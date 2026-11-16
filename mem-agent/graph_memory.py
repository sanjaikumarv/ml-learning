import re
from typing import Literal

from neo4j import GraphDatabase
from openai import OpenAI
from pydantic import BaseModel, Field

EntityType = Literal[
    "Person", "Organization", "Technology", "Location", "Preference", "Role", "Other"
]

RelationType = Literal[
    "HAS_SKILL",
    "WORKS_AT",
    "HAS_ROLE",
    "LIVES_IN",
    "LIKES",
    "DISLIKES",
    "USES",
    "RELATED_TO",
]


class Entity(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    type: EntityType


class Relationship(BaseModel):
    source: str
    target: str
    relation: RelationType


class GraphExtraction(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


class GraphMemory:
    def __init__(self, uri, username, password, database="neo4j"):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )
        self.database = database
        self.client = OpenAI()
        self.driver.verify_connectivity()
        self.create_constraints()

    def create_constraints(self):
        self.driver.execute_query(
            """
            CREATE CONSTRAINT entity_identity IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE (e.user_id, e.key) IS UNIQUE
            """,
            database_=self.database,
        )

    @staticmethod
    def normalize(name):
        return re.sub(r"\s+", " ", name.strip()).casefold()

    def extract(self, text, user_id):
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract factual knowledge graph entities and "
                        "relationships from the USER message only. "
                        "Do not infer unstated facts. "
                        "The user is represented by the entity "
                        f"'USER:{user_id}' of type Person. "
                        "Use that entity as the source for user facts. "
                        "For example: 'I know React' becomes "
                        "USER:<id> HAS_SKILL React. "
                        "For job titles use HAS_ROLE. "
                        "Use only the permitted entity and "
                        "relationship types. "
                        "Every relationship endpoint must be present "
                        "in the entities array. "
                        "Return empty arrays if there are no facts. "
                        "Do not extract questions or instructions "
                        "as personal facts."
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format=GraphExtraction,
            temperature=0,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("Graph extraction returned no parsed data")
        return result

    @staticmethod
    def write_graph(tx, user_id, entities, relationships):
        for entity in entities:
            tx.run(
                """
                MERGE (e:Entity {user_id: $user_id, key: $key})
                ON CREATE SET e.created_at = datetime()
                SET e.name = $name,
                    e.type = $type,
                    e.updated_at = datetime()
                """,
                user_id=user_id,
                key=entity["key"],
                name=entity["name"],
                type=entity["type"],
            ).consume()

        for rel in relationships:
            tx.run(
                """
                MATCH (a:Entity {user_id: $user_id, key: $source})
                MATCH (b:Entity {user_id: $user_id, key: $target})
                MERGE (a)-[r:RELATED_TO {kind: $kind}]->(b)
                ON CREATE SET r.created_at = datetime()
                SET r.updated_at = datetime()
                """,
                user_id=user_id,
                source=rel["source"],
                target=rel["target"],
                kind=rel["relation"],
            ).consume()

    def save(self, user_id, extraction):
        entities = {}
        user_name = f"USER:{user_id}"

        # Trusted identity for the current user
        entities[self.normalize(user_name)] = {
            "key": self.normalize(user_name),
            "name": user_name,
            "type": "Person",
        }

        for entity in extraction.entities:
            key = self.normalize(entity.name)
            if key == self.normalize(user_name):
                continue
            entities[key] = {
                "key": key,
                "name": entity.name.strip(),
                "type": entity.type,
            }

        relationships = []
        for rel in extraction.relationships:
            source = self.normalize(rel.source)
            target = self.normalize(rel.target)

            if source in entities and target in entities and source != target:
                relationships.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": rel.relation,
                    }
                )

        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self.write_graph,
                user_id,
                list(entities.values()),
                relationships,
            )

        return {
            "entities": len(entities),
            "relationships": len(relationships),
        }

    def search(self, user_id, query, limit=20):
        # Extract query entities, without storing them
        extracted = self.extract(query, user_id)

        keys = [
            self.normalize(e.name)
            for e in extracted.entities
            if e.name != f"USER:{user_id}"
        ]

        # Also include user facts for personal-memory questions
        keys.append(self.normalize(f"USER:{user_id}"))

        records, _, _ = self.driver.execute_query(
            """
            MATCH (a:Entity {user_id: $user_id})
                  -[r:RELATED_TO]->(b:Entity {user_id: $user_id})
            WHERE a.key IN $keys OR b.key IN $keys
            RETURN a.name AS source,
                   r.kind AS relation,
                   b.name AS target
            ORDER BY r.updated_at DESC
            LIMIT $limit
            """,
            user_id=user_id,
            keys=list(set(keys)),
            limit=limit,
            database_=self.database,
        )

        return [dict(record) for record in records]

    def close(self):
        self.driver.close()
