import sqlite3
import os
import hashlib

def add_admin(password):
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    hash_object = hashlib.sha512(password.encode())
    hex_dig_key = hash_object.hexdigest()
    cursor.execute(f"INSERT INTO admins (password) VALUES ('{hex_dig_key}')")
    conn.commit()
    conn.close()

def delete_admin(password):
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM admins WHERE password = {password}")
    conn.commit()
    conn.close()

def update_admin(admin_id, password):
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute(f"UPDATE admins SET password = '{password }' WHERE id = {admin_id}")
    conn.commit()
    conn.close()

def verify_admin(admin_id,password):
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    hash_object = hashlib.sha512(password.encode())
    hex_dig_key = hash_object.hexdigest()
    #cursor.execute(f"SELECT id FROM admins WHERE password = '{hex_dig_key}'")
    #select admin_id and password 
    cursor.execute(f"SELECT id FROM admins WHERE id = '{admin_id}' AND password = '{hex_dig_key}'")
    admin = cursor.fetchone()
    conn.close()
    if admin:
        return True
    return None

def delete_table_admin():
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM admins')
    cursor.execute('DELETE FROM sqlite_sequence WHERE name="admins"')
    conn.commit()
    conn.close()

def display_admins():
    dir = os.path.dirname(__file__)
    conn = sqlite3.connect(dir+'/database_user.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id,password FROM admins')
    admins = cursor.fetchall()
    conn.close()
    for admin in admins:
        admin_id, password = admin
        print(f"User ID: {admin_id }, password : {password}")

def main():
    delete_table_admin()
    display_admins()

#create a main

if __name__ == '__main__':
    main()