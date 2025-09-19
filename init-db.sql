-- Initialize database with pgvector extension only. Tables and indexes are created by the app/migrations.
CREATE EXTENSION IF NOT EXISTS vector;

-- You can add indexes after tables are created (via migrations or a separate SQL applied later).
