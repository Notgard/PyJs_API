import os
import sqlite3
import hashlib

def add_user(api_key):
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO users (api_key) VALUES ('{api_key}')")
    conn.commit()
    conn.close()

def delete_user(api_key):
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM users WHERE api_key = {api_key}")
    conn.commit()
    conn.close()

def delete_table_user():
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users')
    cursor.execute('DELETE FROM sqlite_sequence WHERE name="users"')
    conn.commit()
    conn.close()

def update_user(user_id, api_key):
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute(f"UPDATE users SET api_key = '{api_key }' WHERE id = {user_id}")
    conn.commit()
    conn.close()

def verify_user(api_key):
    print(f"api_key: {api_key}, type: {type(api_key)}")
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    hash_object = hashlib.sha512(api_key.encode())
    hex_dig_key = hash_object.hexdigest()
    cursor.execute(f"SELECT id FROM users WHERE api_key = '{hex_dig_key}'")
    user = cursor.fetchone()
    conn.close()
    if user:
        return api_key
    return None

def display_users():
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id,api_key FROM users')
    users = cursor.fetchall()
    conn.close()
    for user in users:
        user_id, api_key = user
        print(f"User ID: {user_id }, API Key : {api_key}")

def main():
    display_users()

#create a main

if __name__ == '__main__':
    main()