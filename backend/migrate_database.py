#!/usr/bin/env python3
"""
Database migration script to add authentication fields to existing database
"""

import sqlite3
import os
from datetime import datetime

def migrate_database():
    """Migrate the existing database to add authentication fields"""
    
    # Database path
    db_path = os.path.join("data", "text_classification.db")
    
    if not os.path.exists(db_path):
        print("Database not found. Creating new database...")
        # If database doesn't exist, just run the normal init
        from models.database import init_db
        init_db()
        print("New database created successfully!")
        return
    
    print(f"Migrating database at: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        users_table_exists = cursor.fetchone() is not None
        
        if users_table_exists:
            # Check if new columns exist
            cursor.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]
            
            print(f"Current users table columns: {columns}")
            
            # Add missing columns
            if 'hashed_password' not in columns:
                print("Adding hashed_password column...")
                cursor.execute("ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255)")
                
            if 'full_name' not in columns:
                print("Adding full_name column...")
                cursor.execute("ALTER TABLE users ADD COLUMN full_name VARCHAR(255)")
                
            if 'is_active' not in columns:
                print("Adding is_active column...")
                cursor.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1")
        else:
            print("Creating users table...")
            cursor.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255),
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Check if classification_results table needs user_id column
        cursor.execute("PRAGMA table_info(classification_results)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'user_id' not in columns:
            print("Adding user_id column to classification_results...")
            cursor.execute("ALTER TABLE classification_results ADD COLUMN user_id INTEGER REFERENCES users(id)")
        
        # Check if csv_processing_jobs table needs user_id column
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='csv_processing_jobs'")
        if cursor.fetchone():
            cursor.execute("PRAGMA table_info(csv_processing_jobs)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'user_id' not in columns:
                print("Adding user_id column to csv_processing_jobs...")
                cursor.execute("ALTER TABLE csv_processing_jobs ADD COLUMN user_id INTEGER REFERENCES users(id)")
        
        # Commit changes
        conn.commit()
        print("Database migration completed successfully!")
        
        # Verify the changes
        cursor.execute("PRAGMA table_info(users)")
        users_columns = [column[1] for column in cursor.fetchall()]
        print(f"Updated users table columns: {users_columns}")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def backup_database():
    """Create a backup of the current database"""
    db_path = os.path.join("data", "text_classification.db")
    if os.path.exists(db_path):
        backup_path = os.path.join("data", f"text_classification_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"Database backed up to: {backup_path}")
        return backup_path
    return None

if __name__ == "__main__":
    print("Starting database migration...")
    
    # Create backup first
    backup_path = backup_database()
    if backup_path:
        print(f"Backup created: {backup_path}")
    
    try:
        migrate_database()
        print("\n✅ Migration completed successfully!")
        print("You can now start the server and use authentication features.")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        if backup_path:
            print(f"You can restore from backup: {backup_path}")
        exit(1)
