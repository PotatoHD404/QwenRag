#!/usr/bin/env python3
"""
Test file for demonstrating Code RAG system capabilities.
This file contains various Python constructs for testing.
"""

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def connect(self):
        """Establish database connection."""
        print(f"Connecting to {self.connection_string}")
        # Connection logic here
        pass
    
    async def execute_query(self, query: str, params=None):
        """Execute a database query with optional parameters."""
        try:
            # Query execution logic
            result = await self._run_query(query, params)
            return result
        except Exception as e:
            print(f"Query failed: {e}")
            raise
    
    async def _run_query(self, query, params):
        """Internal method to run queries."""
        return {"status": "success", "data": []}


def process_data(data_list):
    """Process a list of data items."""
    processed = []
    for item in data_list:
        if isinstance(item, dict):
            processed.append(item.get('value', 0) * 2)
        else:
            processed.append(item * 2)
    return processed


async def main():
    """Main function demonstrating the DatabaseManager."""
    db = DatabaseManager("postgresql://localhost:5432/testdb")
    await db.connect()
    
    result = await db.execute_query("SELECT * FROM users WHERE active = $1", [True])
    print(f"Query result: {result}")
    
    test_data = [1, 2, {'value': 3}, 4]
    processed = process_data(test_data)
    print(f"Processed data: {processed}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 