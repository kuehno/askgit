"""hnsw index

Revision ID: 9333a6579efa
Revises: 5a9df0ee9f9d
Create Date: 2025-10-09 21:49:53.660163

"""

from collections.abc import Sequence

from alembic import op

from askgit.core.config import settings

# revision identifiers, used by Alembic.
revision: str = "9333a6579efa"
down_revision: str | None = "5a9df0ee9f9d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        f"create index if not exists {settings.POSTGRES_SCHEMA}_idx on {settings.POSTGRES_SCHEMA}.github using hnsw (embedding vector_cosine_ops) with (ef_construction=256)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(f"drop index if exists {settings.POSTGRES_SCHEMA}_idx")
