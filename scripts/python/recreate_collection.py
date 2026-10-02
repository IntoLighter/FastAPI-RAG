import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from settings import settings


async def main() -> None:
    engine = create_async_engine(settings.postgres.url)
    async with engine.begin() as connection:
        await connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.execute(
            text(f"TRUNCATE TABLE {settings.postgres.chunk_table}"),
        )
    await engine.dispose()
    print(f"Table {settings.postgres.chunk_table} truncated")


if __name__ == "__main__":
    asyncio.run(main())
