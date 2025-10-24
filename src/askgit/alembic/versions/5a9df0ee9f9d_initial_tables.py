"""initial tables

Revision ID: 5a9df0ee9f9d
Revises: acecc40082bf
Create Date: 2025-10-09 20:42:18.655276

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from askgit.core.config import settings

# revision identifiers, used by Alembic.
revision: str = "5a9df0ee9f9d"
down_revision: str | None = "acecc40082bf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(f"create schema if not exists {settings.POSTGRES_SCHEMA}")

    op.create_table(
        "github",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("content", sa.String, nullable=True),
        sa.Column("embedding", Vector(settings.EMBEDDING_DIMENSIONS), nullable=True),
        sa.Column("repo_url", sa.String, nullable=True),
        sa.Column("meta", sa.JSON, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP),
        sa.Column("updated_at", sa.TIMESTAMP),
        schema=settings.POSTGRES_SCHEMA,
    )

    # Create index on repo_url for efficient querying
    op.create_index(
        "ix_github_repo_url",
        "github",
        ["repo_url"],
        schema=settings.POSTGRES_SCHEMA,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_github_repo_url", table_name="github", schema=settings.POSTGRES_SCHEMA
    )
    op.drop_table("github", schema=settings.POSTGRES_SCHEMA)

    op.execute(f"drop schema if exists {settings.POSTGRES_SCHEMA}")
