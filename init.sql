-- Initialize the piano analysis database
-- This file is run when the PostgreSQL container starts for the first time

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for better performance (tables will be created by SQLAlchemy)
-- These will be applied after the application creates the tables

-- Note: The actual table creation is handled by SQLAlchemy in the backend
-- This file is for any additional database setup needed
