# CREATE TABLE public.embeddings (
#     collection_key character varying,
#     serp_urls      character varying[], -- Assuming ARRAY of character varying
#     created_at     timestamp without time zone,
#     updated_at     timestamp without time zone
# );

import psycopg2

import os
from dotenv import load_dotenv
load_dotenv()

dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")

def create_connection():
    try:
        connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        cursor = connection.cursor()
        return connection, cursor
    except psycopg2.Error as error:
        print("Error creating database connection:", error)
        return None, None

def close_connection(connection):
    if connection:
        connection.close()
        print("Connection closed.")

def create_record(collection_key, serp_urls):
    connection, cursor = create_connection()
    if connection and cursor:
        try:
            insert_query = """
            INSERT INTO embeddings (collection_key, serp_urls)
            VALUES (%s, %s);
            """
            cursor.execute(insert_query, (collection_key, serp_urls))
            connection.commit()
            print("Record created successfully.")
        except psycopg2.Error as error:
            print("Error with database operation:", error)
        finally:
            close_connection(connection)

def update_record(collection_key, new_serp_urls):
    connection, cursor = create_connection()

    if connection and cursor:
        try:
            # SQL query to update the serp_urls for a specific record
            update_query = """
            UPDATE embeddings
            SET serp_urls = %s, updated_at = CURRENT_TIMESTAMP
            WHERE collection_key = %s;
            """
            cursor.execute(update_query, (new_serp_urls, collection_key))
            connection.commit()
            print("Record updated successfully.")
        except psycopg2.Error as error:
            print("Error with database operation:", error)
        finally:
            close_connection(connection)
