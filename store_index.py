import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embedding
import config

class PineconeHandler:
    def __init__(self, index_name="new_info", dimension=384, metric="cosine", cloud="aws", region="us-east-1"):
        load_dotenv()
        
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Load embeddings
        self.embeddings = download_hugging_face_embedding()

    def index_exists(self):
        """Checks if the Pinecone index already exists."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        print(config.get_chatbot_name())
        return self.index_name in existing_indexes

    def namespace_exists(self, namespace):
        """Checks if a given namespace exists in Pinecone."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            existing_namespaces = stats.get("namespaces", {})

            if namespace in existing_namespaces:
                print(f"✅ Namespace '{namespace}' exists.")
                return True
            else:
                print(f"❌ Namespace '{namespace}' does NOT exist.")
                return False
        except Exception as e:
            print(f"⚠️ Error checking namespace: {e}")
            return False
        

    def create_index(self):
        """Creates a Pinecone index only if it doesn't already exist."""
        if self.index_exists():
            print(f"⚡ Index '{self.index_name}' already exists. Skipping creation.")
            return
        
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )
        print(f"✅ Created new Pinecone index: {self.index_name}")
    
    def upsert_documents(self, data_path="Data/upload/"):
        """Loads PDF data, splits it, and upserts into Pinecone only if the index is new."""
        # if self.index_exists():
        #     print(f"⚡ Index '{self.index_name}' already exists. Using existing data.")
        #     return PineconeVectorStore.from_existing_index(
        #         index_name=self.index_name,
        #         embedding=self.embeddings
        #     )

        # If index doesn't exist, create embeddings and store them
        print(f"🚀 Index '{self.index_name}' is new. Uploading documents...")
        extracted_data = load_pdf_file(data=data_path)
        text_chunks = text_split(extracted_data=extracted_data)

        doc_search = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace = f'user_id_{config.get_chatbot_name()}'

        )

        print(f"✅ Documents uploaded to Pinecone index '{self.index_name}'")
        return doc_search
