# From original SQLNet code.
# Wonseok modified. 20180607

import records
import re
from babel.numbers import parse_decimal, NumberFormatError


schema_re = re.compile(r'\((.+)\)')  # group (.......) dfdf (.... )group
# ? zero or one time appear of preceding character, * zero or several time appear of preceding character.
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')
# Catch something like -34.34, .4543,
# | is 'or'

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']


class DBEngine:

    def __init__(self, pdb="postgres://postgres:postgres@localhost:5432/honda_dev"):
        self.pdb = records.Database(pdb)

    def execute_query(self, table_id, query, columns, types,table, *args, **kwargs):
        print("EXECUTING QUERY")
        # return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, columns, types, *args, **kwargs)
        return self.generateDBSQL(table_id, query.sel_index, query.agg_index, query.conditions, columns, types, table, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, columns, types, table, lower=True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))

        print("DBENGNINE", columns)
        print(table_id)
        print(select_index)
        print(aggregation_index)
        print(conditions)

        table_info = self.pdb.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[
            0].sql.replace('\n', '')
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str) or isinstance(val, str)):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val, locale='en_US'))

                except NumberFormatError as e:
                    try:
                        # need to understand and debug this part.
                        val = float(num_re.findall(val)[0])
                        print(e)
                    except:
                        pass
            where_clause.append('lower(col{}) {} lower(:col{})'.format(
                col_index, cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(
            select, table_id, where_str)

        print(query)
        print(self.generateDBSQL("trips", select_index,
                                 aggregation_index, conditions, columns, types, lower=True))

        out = self.pdb.query(query, **where_map)

        return [o.result for o in out]

    def generateDBSQL(self, table_id, select_index, aggregation_index, conditions, columns, types, table, lower=True):
        # schema_str = schema_re.findall(table_info)[0]
        # schema = {}
        # for tup in schema_str.split(', '):
        #     c, t = tup.split()
        #     schema[c] = t

        print("Generating")
        print("SQLENGINIE", columns)
        print(table_id)
        print(table)
        print(select_index)
        print(aggregation_index)
        print(conditions)

        select = columns[select_index]

        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        for col_index, op, val in conditions:
            print(types[col_index])
            col = columns[col_index]
            if isinstance(val, str):
                val = "'{}'".format(val.lower())
                col = "LOWER({})".format(col)
            if types[col_index] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(
                        re.search(r'\d+', val).group(), locale='en_US'))
                except NumberFormatError as e:
                    try:
                        # need to understand and debug this part.
                        val = float(num_re.findall(val)[0])
                        print(e)
                    except:
                        # Although column is of number, selected one is not number. Do nothing in this case.
                        pass

            where_clause.append('{} {} {}'.format(
                col, cond_ops[op], val))

        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)

        print(select)
        if("id" in select.lower()):
            print("REPLACE WITH ALL", select)
            print(columns)
        query = 'SELECT {} AS result FROM {} {}'.format(
            select, table, where_str)
        print(query)

        out = self.pdb.query(query)
        print("QUERY SUCCESFUL")
        return [o.result for o in out]

    def execute_return_query(self, table_id, select_index, aggregation_index, conditions, lower=True):
        print("EXECUTING RETURN QUERY")
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = self.pdb.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[
            0].sql.replace('\n', '')
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str) or isinstance(val, str)):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    # print('!!!!!!value of val is: ', val, 'type is: ', type(val))
                    # val = float(parse_decimal(val)) # somehow it generates error.
                    val = float(parse_decimal(val, locale='en_US'))
                    # print('!!!!!!After: val', val)

                except NumberFormatError as e:
                    print(e)
                    val = float(num_re.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(
                col_index, cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(
            select, table_id, where_str)
        # print query
        out = self.pdb.query(query, **where_map)

        return [o.result for o in out], query

    def show_table(self, table_id):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        rows = self.pdb.query('select * from ' + table_id)
        print(rows.dataset)
