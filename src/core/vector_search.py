import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import logging

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class VectorSearchEngine:
    """Vector search engine for intervention database using ChromaDB and sentence-transformers"""
    
    def __init__(self, persist_directory: str = "./data/embeddings/chromadb"):
        self.persist_directory = persist_directory
        self.collection_name = "interventions"
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and embedding model"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Initialize embedding model
            model_name = config.get('model.embedding.name', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Road safety interventions database"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector search engine: {e}")
            raise
    
    def generate_embeddings(self, interventions: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for intervention descriptions"""
        try:
            # Extract text for embedding
            texts = []
            for intervention in interventions:
                # Combine multiple text fields for better semantic representation
                text_parts = [
                    intervention.get('intervention_name', ''),
                    intervention.get('description', ''),
                    intervention.get('category', ''),
                    intervention.get('problem_type', ''),
                    ' '.join(intervention.get('compliance_requirements', []))
                ]
                combined_text = ' '.join(filter(None, text_parts))
                texts.append(combined_text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def populate_database(self, interventions: List[Dict[str, Any]]):
        """Populate ChromaDB with interventions and their embeddings"""
        try:
            logger.info(f"Populating database with {len(interventions)} interventions...")
            
            # Generate embeddings
            embeddings = self.generate_embeddings(interventions)
            
            # Prepare data for ChromaDB
            ids = [intervention['intervention_id'] for intervention in interventions]
            documents = []
            metadatas = []
            
            for intervention in interventions:
                # Create document text
                doc_text = f"{intervention['intervention_name']}. {intervention['description']}"
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    'category': intervention['category'],
                    'problem_type': intervention['problem_type'],
                    'cost_total': intervention['cost_estimate']['total'],
                    'impact_percent': intervention['predicted_impact']['accident_reduction_percent'],
                    'confidence': intervention['predicted_impact']['confidence_level'],
                    'timeline': intervention['implementation_timeline']
                }
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info("Database populated successfully!")
            
        except Exception as e:
            logger.error(f"Failed to populate database: {e}")
            raise
    
    def search(self, query: str, n_results: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant interventions
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of relevant interventions with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=filters
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'intervention_id': results['ids'][0][i],
                    'score': results['distances'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_category(self, query: str, category: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search within a specific category"""
        filters = {"category": category}
        return self.search(query, n_results, filters)
    
    def search_by_cost_range(self, query: str, min_cost: int, max_cost: int, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search within a cost range"""
        # Note: ChromaDB doesn't support range queries directly, so we'll filter after search
        results = self.search(query, n_results * 2)  # Get more results to filter
        
        filtered_results = []
        for result in results:
            cost = result['metadata']['cost_total']
            if min_cost <= cost <= max_cost:
                filtered_results.append(result)
                if len(filtered_results) >= n_results:
                    break
        
        return filtered_results
    
    def get_similar_interventions(self, intervention_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find interventions similar to a given intervention"""
        try:
            # Get the intervention's embedding
            intervention_data = self.collection.get(ids=[intervention_id])
            if not intervention_data['embeddings']:
                return []
            
            embedding = intervention_data['embeddings'][0]
            
            # Search for similar interventions
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results + 1  # +1 to exclude the original
            )
            
            # Format results (excluding the original intervention)
            formatted_results = []
            for i in range(len(results['ids'][0])):
                if results['ids'][0][i] != intervention_id:
                    result = {
                        'intervention_id': results['ids'][0][i],
                        'score': results['distances'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error(f"Failed to get similar interventions: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_interventions': count,
                'collection_name': self.collection_name,
                'embedding_model': config.get('model.embedding.name', 'sentence-transformers/all-MiniLM-L6-v2'),
                'embedding_dimension': config.get('model.embedding.dimension', 384)
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

class InterventionRetriever:
    """High-level interface for intervention retrieval"""
    
    def __init__(self, vector_engine: VectorSearchEngine, interventions_data: List[Dict[str, Any]]):
        self.vector_engine = vector_engine
        self.interventions_data = {intv['intervention_id']: intv for intv in interventions_data}
    
    def retrieve_interventions(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant interventions with full data"""
        # Get vector search results
        search_results = self.vector_engine.search(query, n_results)
        
        # Enrich with full intervention data
        enriched_results = []
        for result in search_results:
            intervention_id = result['intervention_id']
            if intervention_id in self.interventions_data:
                intervention = self.interventions_data[intervention_id].copy()
                intervention['relevance_score'] = 1 - result['score']  # Convert distance to similarity
                enriched_results.append(intervention)
        
        return enriched_results
    
    def retrieve_by_category(self, query: str, category: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve interventions from specific category"""
        search_results = self.vector_engine.search_by_category(query, category, n_results)
        
        enriched_results = []
        for result in search_results:
            intervention_id = result['intervention_id']
            if intervention_id in self.interventions_data:
                intervention = self.interventions_data[intervention_id].copy()
                intervention['relevance_score'] = 1 - result['score']
                enriched_results.append(intervention)
        
        return enriched_results
    
    def retrieve_by_cost_effectiveness(self, query: str, max_cost: int = 100000) -> List[Dict[str, Any]]:
        """Retrieve cost-effective interventions"""
        # Get more results to filter by cost-effectiveness
        search_results = self.vector_engine.search(query, 20)
        
        enriched_results = []
        for result in search_results:
            intervention_id = result['intervention_id']
            if intervention_id in self.interventions_data:
                intervention = self.interventions_data[intervention_id].copy()
                intervention['relevance_score'] = 1 - result['score']
                
                # Calculate cost-effectiveness score
                cost = intervention['cost_estimate']['total']
                impact = intervention['predicted_impact']['accident_reduction_percent']
                if cost > 0:
                    intervention['cost_effectiveness'] = impact / (cost / 1000)  # Impact per 1000 INR
                else:
                    intervention['cost_effectiveness'] = 0
                
                if cost <= max_cost:
                    enriched_results.append(intervention)
        
        # Sort by cost-effectiveness
        enriched_results.sort(key=lambda x: x['cost_effectiveness'], reverse=True)
        return enriched_results[:10]

def initialize_vector_search(interventions_file: str = "data/interventions/interventions.json") -> InterventionRetriever:
    """Initialize vector search system"""
    try:
        # Load interventions data
        with open(interventions_file, 'r', encoding='utf-8') as f:
            interventions = json.load(f)
        
        # Initialize vector search engine
        vector_engine = VectorSearchEngine()
        
        # Check if database is already populated
        stats = vector_engine.get_collection_stats()
        if stats.get('total_interventions', 0) == 0:
            logger.info("Populating vector database...")
            vector_engine.populate_database(interventions)
        else:
            logger.info(f"Vector database already populated with {stats['total_interventions']} interventions")
        
        # Create retriever
        retriever = InterventionRetriever(vector_engine, interventions)
        
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to initialize vector search: {e}")
        raise
