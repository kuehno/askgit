"""enable extension

Revision ID: acecc40082bf
Revises:
Create Date: 2025-10-09 20:42:09.986926

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "acecc40082bf"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("create extension vector")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("drop extension vector")
