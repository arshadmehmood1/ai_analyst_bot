from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config.settings import settings

class DatabaseConnection:
    """
    Handles SQL database connection using SQLAlchemy.
    This will be used when you later connect your complaint system
    to a live database instead of CSV.
    """

    def __init__(self, db_url: str = None):
        self.db_url = db_url or settings.DB_URL

        if not self.db_url:
            raise ValueError("DB_URL is not set in the .env file.")

        # Create engine
        self.engine = create_engine(self.db_url, echo=False)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def get_session(self):
        """
        Opens a new database session
        """
        try:
            session = self.SessionLocal()
            return session
        except Exception as e:
            raise RuntimeError(f"Error creating DB session: {str(e)}")


# Optional helper function
def get_db():
    """FastAPI-style generator for DB dependency"""
    db = DatabaseConnection().get_session()
    try:
        yield db
    finally:
        db.close()