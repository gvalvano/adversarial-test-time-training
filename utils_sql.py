import sqlite3
import datetime


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file. If you pass the file name as ':memory:', it will create a new database that resides
                    in the memory (RAM) instead of a database file on disk.
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def alter_table(conn, alter_table_sql):
    """ Alter a table from the create_table_sql statement
    :param conn: Connection object
    :param alter_table_sql: a ALTER TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(alter_table_sql)
    except sqlite3.Error as e:
        print(e)


def add_new_column(conn, table_name, column_name, column_type):
    """ Alter a table from the create_table_sql statement
    :param conn: Connection object
    :param table_name: name of the table in the database
    :param column_name: the name of the column to add
    :param column_type: the type of the column to add
    :return:
    """
    sql_command = "ALTER TABLE {0} ADD COLUMN {1} {2}".format(table_name, column_name, column_type)
    try:
        cur = conn.cursor()
        cur.execute(sql_command)
    except sqlite3.Error as e:
        print(e)


def update_table(conn, table_name, column_name, value, condition):
    """ Alter a table from the create_table_sql statement
    :param conn: Connection object
    :param table_name: name of the table in the database
    :param column_name: column to update
    :param value: value to insert
    :param condition: a string imposing conditions on the query
    :return:
    """
    cond = '' if condition is None else ' WHERE {0}'.format(condition)
    sql_command = "UPDATE {0} SET {1}={2}{3}".format(table_name, column_name, value, cond)
    try:
        cur = conn.cursor()
        cur.execute(sql_command)
    except sqlite3.Error as e:
        print(e)


def delete_record(conn, table_name, condition):
    """ Delete a record from a table
    :param conn: Connection object
    :param table_name: name of the table in the database
    :param condition: a string imposing conditions on the query
    :return:
    """
    sql_command = "DELETE FROM {0} WHERE {1}".format(table_name, condition)
    try:
        cur = conn.cursor()
        cur.execute(sql_command)
    except sqlite3.Error as e:
        print(e)


def delete_column(conn, table_name, column_name):
    """ Alter a table from the create_table_sql statement
    :param conn: Connection object
    :param table_name: name of the table in the database
    :param column_name: the name of the column to delete
    :return:
    """
    sql_command = "ALTER TABLE {0} DROP COLUMN {1}".format(table_name, column_name)
    try:
        cur = conn.cursor()
        cur.execute(sql_command)
    except sqlite3.Error as e:
        print(e)


def insert_new_record(conn, table_name, values):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param table_name: name of the table in the database
    :param values: values to insert
    :return:
    """
    assert isinstance(values, list)
    sql = """INSERT INTO {0}(RUN_ID, PERC, SPLIT, CONFIG, EXPERIMENT_TYPE, DATASET_NAME, INPUT_SIZE, EPOCH,
            AVERAGE_DICE, STD_DICE, AVERAGE_DICE_PER_CLASS, STD_DICE_PER_CLASS, DICE_VALUES, DICE_VALUES_PER_CLASS, 
            AVERAGE_IOU, STD_IOU, AVERAGE_IOU_PER_CLASS, STD_IOU_PER_CLASS, IOU_VALUES, IOU_VALUES_PER_CLASS, 
            AVERAGE_HD, STD_HD, AVERAGE_HD_PER_CLASS, STD_HD_PER_CLASS, HD_VALUES, HD_VALUES_PER_CLASS, TIMESTAMP) 
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""".format(table_name)
    cur = conn.cursor()
    time_stamp = datetime.datetime.now().strftime("%Y-%b-%d, %A %I:%M:%S")
    values = values + [time_stamp]
    cur.execute(sql, values)
    return cur.lastrowid  # return the generated id


def query_table(conn, table_name, values, distinct=False, condition=None, order=None):
    """ Query the table inside a dataset connected through conn.
    :param conn: Connection object
    :param table_name: name of the table in the database
    :param values: values to query
    :param distinct: if true, remove duplicates
    :param condition: a string imposing conditions on the query
    :param order: a string for ordering the query results (by a specific column)
    :return: query results

    Examples:
        values = "dice, iou, hd"
        table = "T_Metrics"
        condition = "percentage == 25 AND run_id LIKE 'My_Great_Model_%'"
        results = query_table(conn, table, values, condition)
    """
    cond = '' if condition is None else ' WHERE {0}'.format(condition)
    order = '' if order is None else ' ORDER BY {0} ASC'.format(order)
    values = values if not distinct else 'DISTINCT {0}'.format(values)

    sql_command = "SELECT {0} FROM {1}{2}{3}".format(values, table_name, cond, order)
    cur = conn.cursor()
    cur.execute(sql_command)
    rows = cur.fetchall()
    return rows


def create_table_statement(table_name):
    statement = """ CREATE TABLE IF NOT EXISTS {0} (
                        id integer PRIMARY KEY,
                        RUN_ID text NOT NULL,
                        PERC text NOT NULL,
                        SPLIT text NOT NULL,
                        CONFIG text NOT NULL,
                        EXPERIMENT_TYPE text NOT NULL,
                        DATASET_NAME text NOT NULL,
                        INPUT_SIZE text NOT NULL,
                        EPOCH integer NOT NULL,
                        AVERAGE_DICE float NOT NULL,
                        STD_DICE float NOT NULL,
                        AVERAGE_DICE_PER_CLASS text NOT NULL,
                        STD_DICE_PER_CLASS text NOT NULL,
                        DICE_VALUES text NOT NULL,
                        DICE_VALUES_PER_CLASS text NOT NULL,
                        AVERAGE_IOU float NOT NULL,
                        STD_IOU float NOT NULL,
                        AVERAGE_IOU_PER_CLASS text NOT NULL,
                        STD_IOU_PER_CLASS text NOT NULL,
                        IOU_VALUES text NOT NULL,
                        IOU_VALUES_PER_CLASS text NOT NULL,
                        AVERAGE_HD float NOT NULL,
                        STD_HD float NOT NULL,
                        AVERAGE_HD_PER_CLASS text NOT NULL,
                        STD_HD_PER_CLASS text NOT NULL,
                        HD_VALUES text NOT NULL,
                        HD_VALUES_PER_CLASS text NOT NULL,
                        TIMESTAMP text
                        ); """.format(table_name)
    return statement


def add_db_entry(entries, table_name, database='test_results_by_patient.db'):
    """Insert values in results SQL database. """
    # create a database connection
    conn = create_connection(database)
    # table_name = args.table_name

    # create tables
    if conn is not None:
        statement = """ CREATE TABLE IF NOT EXISTS {0} (
                        id integer PRIMARY KEY,
                        RUN_ID text NOT NULL,
                        PERC text NOT NULL,
                        SPLIT text NOT NULL,
                        CONFIG text NOT NULL,
                        EXPERIMENT_TYPE text NOT NULL,
                        DATASET_NAME text NOT NULL,
                        INPUT_SIZE text NOT NULL,
                        EPOCH integer NOT NULL,
                        AVERAGE_DICE float NOT NULL,
                        STD_DICE float NOT NULL,
                        AVERAGE_DICE_PER_CLASS text NOT NULL,
                        STD_DICE_PER_CLASS text NOT NULL,
                        DICE_VALUES text NOT NULL,
                        DICE_VALUES_PER_CLASS text NOT NULL,
                        AVERAGE_IOU float NOT NULL,
                        STD_IOU float NOT NULL,
                        AVERAGE_IOU_PER_CLASS text NOT NULL,
                        STD_IOU_PER_CLASS text NOT NULL,
                        IOU_VALUES text NOT NULL,
                        IOU_VALUES_PER_CLASS text NOT NULL,
                        TIMESTAMP text
                    ); """.format(table_name)
        create_table(conn, statement)
    else:
        print("\033[31m  Error! cannot create the database connection.\033[0m")
        raise

    with conn:
        # get values and insert into table:
        values = [entries['run_id'], entries['n_sup_vols'], entries['split_number'], str(entries['config']),
                  entries['experiment_type'], entries['dataset_name'],
                  str(entries['input_size']),
                  int(entries['epoch']),
                  float(entries['avg_dice']),
                  float(entries['std_dice']),
                  str(entries['avg_dice_per_class']),
                  str(entries['std_dice_per_class']),
                  str(entries['dice_list']),
                  str(entries['dice_list_per_class']),
                  float(entries['avg_iou']),
                  float(entries['std_iou']),
                  str(entries['avg_iou_per_class']),
                  str(entries['std_iou_per_class']),
                  str(entries['iou_list']),
                  str(entries['iou_list_per_class'])
                  ]
        insert_new_record(conn, table_name, values)
