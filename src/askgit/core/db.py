from sqlmodel import create_engine

from askgit.core.config import settings

engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))
