from datetime import datetime

import pytz
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlmodel import Column, Field, MetaData, SQLModel

tz = pytz.timezone("Europe/Berlin")


class Github(SQLModel, table=True):
    metadata = MetaData(schema="knowledge")
    __tablename__ = "github"

    id: int = Field(sa_column=Column(sa.BigInteger, primary_key=True))
    repo_url: str | None = Field(default=None, index=True)
    content: str | None = None
    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector))
    meta: dict | None = Field(default=None, sa_column=Column(sa.JSON))
    created_at: datetime = Field(
        sa_column=Column(sa.DateTime(), onupdate=datetime.now(tz))
    )
    updated_at: datetime | None = Field(
        sa_column=Column(sa.DateTime(), onupdate=datetime.now(tz))
    )
