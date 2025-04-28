from neo4j import GraphDatabase
from py2neo import Graph
import logging

class TwitterGraph:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.graph = Graph(uri, auth=(user, password))
            self.connected = True
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j database. Some features may be limited. Error: {str(e)}")
            self.connected = False
        self.logger = logging.getLogger(__name__)

    def close(self):
        if self.connected:
            self.driver.close()

    def initialize_schema(self):
        """Initialize the Neo4j schema with constraints"""
        if not self.connected:
            return
        with self.driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            session.run("CREATE CONSTRAINT tweet_id IF NOT EXISTS FOR (t:Tweet) REQUIRE t.id IS UNIQUE")
            session.run("CREATE CONSTRAINT hashtag_text IF NOT EXISTS FOR (h:Hashtag) REQUIRE h.text IS UNIQUE")

    def create_user(self, user_id, username):
        """Create or update a user node"""
        query = """
        MERGE (u:User {id: $user_id})
        SET u.username = $username
        RETURN u
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id, username=username)

    def create_tweet(self, tweet_id, text, sentiment_score, user_id):
        """Create a tweet node and connect it to the user"""
        query = """
        MATCH (u:User {id: $user_id})
        MERGE (t:Tweet {id: $tweet_id})
        SET t.text = $text, t.sentiment_score = $sentiment_score
        MERGE (u)-[:TWEETED]->(t)
        RETURN t
        """
        with self.driver.session() as session:
            session.run(query, 
                       tweet_id=tweet_id,
                       text=text,
                       sentiment_score=sentiment_score,
                       user_id=user_id)

    def add_hashtags(self, tweet_id, hashtags):
        """Add hashtags to a tweet"""
        query = """
        MATCH (t:Tweet {id: $tweet_id})
        UNWIND $hashtags AS hashtag
        MERGE (h:Hashtag {text: hashtag})
        MERGE (t)-[:CONTAINS]->(h)
        """
        with self.driver.session() as session:
            session.run(query, tweet_id=tweet_id, hashtags=hashtags)

    def create_user_relationships(self):
        """Create relationships between users based on interactions"""
        query = """
        MATCH (u1:User)-[:TWEETED]->(t:Tweet)-[:MENTIONS]->(u2:User)
        WHERE u1 <> u2
        MERGE (u1)-[r:INTERACTS_WITH]->(u2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1
        """
        with self.driver.session() as session:
            session.run(query)

    def find_similar_users(self, user_id, limit=10):
        """Find users with similar interaction patterns"""
        query = """
        MATCH (u1:User {id: $user_id})-[:INTERACTS_WITH]->(u2:User)
        MATCH (u2)-[:INTERACTS_WITH]->(u3:User)
        WHERE u1 <> u3
        RETURN u3.id as user_id, u3.username as username, count(*) as similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id, limit=limit)
            return [dict(record) for record in result]

    def analyze_hashtag_relationships(self):
        """Analyze co-occurrence of hashtags"""
        query = """
        MATCH (h1:Hashtag)<-[:CONTAINS]-(t:Tweet)-[:CONTAINS]->(h2:Hashtag)
        WHERE h1 <> h2
        MERGE (h1)-[r:CO_OCCURS_WITH]->(h2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1
        """
        with self.driver.session() as session:
            session.run(query)

    def get_user_recommendations(self, user_id, limit=10):
        """Get recommendations for a user"""
        query = """
        MATCH (u1:User {id: $user_id})-[:INTERACTS_WITH]->(u2:User)
        MATCH (u2)-[:INTERACTS_WITH]->(u3:User)
        WHERE u1 <> u3 AND NOT (u1)-[:INTERACTS_WITH]->(u3)
        RETURN u3.id as user_id, u3.username as username, count(*) as recommendation_score
        ORDER BY recommendation_score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id, limit=limit)
            return [dict(record) for record in result]

    def get_hashtag_recommendations(self, user_id, limit=10):
        """Get hashtag recommendations for a user"""
        query = """
        MATCH (u:User {id: $user_id})-[:TWEETED]->(t:Tweet)-[:CONTAINS]->(h:Hashtag)
        MATCH (h)-[r:CO_OCCURS_WITH]->(h2:Hashtag)
        WHERE NOT (t)-[:CONTAINS]->(h2)
        RETURN h2.text as hashtag, sum(r.weight) as recommendation_score
        ORDER BY recommendation_score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, user_id=user_id, limit=limit)
            return [dict(record) for record in result]

    def detect_communities(self):
        """Detect communities using Louvain algorithm"""
        query = """
        CALL gds.louvain.stream({
            nodeQuery: 'MATCH (u:User) RETURN id(u) as id',
            relationshipQuery: 'MATCH (u1:User)-[r:INTERACTS_WITH]->(u2:User) 
                              RETURN id(u1) as source, id(u2) as target, r.weight as weight',
            relationshipWeightProperty: 'weight'
        })
        YIELD nodeId, communityId
        MATCH (u:User) WHERE id(u) = nodeId
        RETURN u.id as user_id, u.username as username, communityId
        ORDER BY communityId
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result] 