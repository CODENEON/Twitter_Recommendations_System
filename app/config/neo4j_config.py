import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jConfig:
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    
    @classmethod
    def get_connection_params(cls):
        return {
            'uri': cls.NEO4J_URI,
            'user': cls.NEO4J_USER,
            'password': cls.NEO4J_PASSWORD
        } 