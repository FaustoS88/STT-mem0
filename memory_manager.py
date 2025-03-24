"""
Memory Management Tool for AI Companion
---------------------------------------
This script allows you to view and manage memories from the PostgreSQL database,
giving you control over what your AI companions remember.
"""

import psycopg2
from termcolor import cprint
import json

def inspect_table_structure(cursor, table_name):
    """Inspect the structure of a table to understand its columns"""
    try:
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")
        columns = cursor.fetchall()
        
        cprint(f"\n📋 Table structure for '{table_name}':", "cyan")
        for col_name, data_type in columns:
            cprint(f"  - {col_name} ({data_type})", "cyan")
        
        return [col[0] for col in columns]
    except Exception as e:
        cprint(f"❌ Error inspecting table structure: {e}", "red")
        return []

def view_memories(cursor, columns, limit=10, offset=0, show_vectors=False):
    """View memories with pagination based on actual table structure"""
    try:
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM mem0")
        total = cursor.fetchone()[0]
        
        if total == 0:
            cprint("ℹ️ No memories found in the database.", "blue")
            return 0, 0, 0
        
        # Build a query based on available columns
        query = f"SELECT * FROM mem0 ORDER BY id DESC LIMIT {limit} OFFSET {offset}"
        cursor.execute(query)
        memories = cursor.fetchall()
        
        cprint(f"\n📚 Showing memories {offset+1}-{min(offset+limit, total)} of {total}:", "cyan")
        cprint("=" * 80, "cyan")
        
        for i, memory in enumerate(memories):
            cprint(f"Memory #{i+1+offset}:", "yellow")
            
            # Display each column and its value
            for j, col_name in enumerate(columns):
                value = memory[j]
                
                # Skip vector display unless specifically requested
                if col_name == "vector" and not show_vectors:
                    cprint(f"  {col_name}: [vector data hidden - use 'v' to view vectors]", "white")
                    continue
                
                # Try to parse JSON if the value looks like JSON
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    try:
                        parsed = json.loads(value)
                        value = json.dumps(parsed, indent=2)
                    except:
                        pass
                elif col_name == "payload" and hasattr(value, 'get'):
                    # Format the payload nicely if it's already a dictionary
                    try:
                        # Extract the most relevant information
                        data = value.get('data', 'No data')
                        user_id = value.get('user_id', 'No user ID')
                        created_at = value.get('created_at', 'No timestamp')
                        
                        cprint(f"  {col_name}:", "white")
                        cprint(f"    Content: {data}", "green")
                        cprint(f"    User: {user_id}", "white")
                        cprint(f"    Created: {created_at}", "white")
                        continue
                    except:
                        pass
                
                # Truncate long values
                if isinstance(value, str) and len(value) > 500:
                    value = value[:500] + "... (truncated)"
                
                cprint(f"  {col_name}: {value}", "white")
            
            cprint("-" * 80, "cyan")
        
        return total, offset, limit
    
    except Exception as e:
        cprint(f"❌ Error viewing memories: {e}", "red")
        return 0, 0, 0

def delete_specific_memory(cursor, conn, memory_id):
    """Delete a specific memory by ID"""
    try:
        # Always treat the memory_id as a string for UUID compatibility
        cursor.execute("DELETE FROM mem0 WHERE id::text = %s", (str(memory_id),))
        rows_deleted = cursor.rowcount
        conn.commit()
        
        if rows_deleted > 0:
            cprint(f"✅ Successfully deleted memory #{memory_id}", "green")
        else:
            cprint(f"❌ Memory #{memory_id} not found", "red")
    
    except Exception as e:
        cprint(f"❌ Error deleting memory: {e}", "red")
        # Rollback the transaction to avoid "transaction is aborted" errors
        conn.rollback()

def manage_memories():
    # Database connection parameters (same as in the companion scripts)
    db_params = {
        "user": "postgres",
        "password": "postgres",
        "host": "127.0.0.1",
        "port": "54329",
        "database": "postgres"
    }
    
    try:
        # Connect to the database
        cprint("🔌 Connecting to PostgreSQL database...", "yellow")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Check if the mem0 table exists
        cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'mem0')")
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            cprint("❌ Memory table 'mem0' doesn't exist yet. Nothing to manage.", "red")
            return
        
        # Get the current count of memories
        cursor.execute("SELECT COUNT(*) FROM mem0")
        count = cursor.fetchone()[0]
        cprint(f"📊 Current memory count in mem0 table: {count}", "cyan")
        
        if count == 0:
            cprint("ℹ️ No memories to manage.", "blue")
            return
        
        # Inspect table structure
        columns = inspect_table_structure(cursor, "mem0")
        
        # Interactive menu
        offset = 0
        limit = 5
        show_vectors = False
        
        while True:
            cprint("\n🔍 Memory Management Options:", "magenta")
            cprint("1. View memories", "yellow")
            cprint("2. Delete specific memory", "yellow")
            cprint("3. Delete all memories", "yellow")
            cprint("4. Toggle vector display", "yellow")
            cprint("5. Exit", "yellow")
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                total, offset, limit = view_memories(cursor, columns, limit, offset, show_vectors)
                
                if total > limit:
                    nav = input("\nNavigation (n: next page, p: previous page, b: back to menu): ").strip().lower()
                    if nav == 'n' and offset + limit < total:
                        offset += limit
                    elif nav == 'p' and offset - limit >= 0:
                        offset -= limit
            
            elif choice == "2":
                memory_id = input("Enter the ID of the memory to delete: ").strip()
                try:
                    memory_id = int(memory_id) if memory_id.isdigit() else memory_id
                    delete_specific_memory(cursor, conn, memory_id)
                except ValueError:
                    cprint("❌ Invalid ID. Please enter a valid memory ID.", "red")
            
            elif choice == "3":
                confirm = input("⚠️ Are you sure you want to delete ALL memories? (y/n): ").strip().lower()
                if confirm == 'y':
                    cursor.execute("DELETE FROM mem0")
                    conn.commit()
                    cprint(f"✅ Successfully deleted all {count} memories!", "green")
                    
                    # Verify the deletion
                    cursor.execute("SELECT COUNT(*) FROM mem0")
                    new_count = cursor.fetchone()[0]
                    cprint(f"📊 New memory count: {new_count}", "cyan")
                else:
                    cprint("🛑 Deletion cancelled.", "blue")
            
            elif choice == "4":
                show_vectors = not show_vectors
                cprint(f"ℹ️ Vector display is now {'ON' if show_vectors else 'OFF'}", "blue")
            
            elif choice == "5":
                cprint("👋 Exiting memory management.", "blue")
                break
            
            else:
                cprint("❌ Invalid choice. Please enter a number between 1 and 5.", "red")
        
    except Exception as e:
        cprint(f"❌ Error: {e}", "red")
    finally:
        if 'conn' in locals():
            cursor.close()
            conn.close()
            cprint("🔌 Database connection closed.", "yellow")

if __name__ == "__main__":
    cprint("🧠 AI Companion Memory Management Tool", "magenta")
    cprint("--------------------------------------", "magenta")
    manage_memories()
    cprint("\n🚀 Run either aiVoiceCompanion.py or aiTextCompanion.py to continue your conversation.", "green")