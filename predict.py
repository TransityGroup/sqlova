#!/usr/bin/env python

# Use existing model to predict sql from tables and questions.
#
# For example, you can get a pretrained model from https://github.com/naver/sqlova/releases:
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_bert_best.pt
#    https://github.com/naver/sqlova/releases/download/SQLova-parameters/model_best.pt
#
# Make sure you also have the following support files (see README for where to get them):
#    - bert_config_uncased_*.json
#    - vocab_uncased_*.txt
#
# Finally, you need some data - some files called:
#    - <split>.db
#    - <split>.jsonl
#    - <split>.tables.jsonl
#    - <split>_tok.jsonl         # derived using annotate_ws.py
# You can play with the existing train/dev/test splits, or make your own with
# the add_csv.py and add_question.py utilities.
#
# Once you have all that, you are ready to predict, using:
#   python predict.py \
#     --bert_type_abb uL \       # need to match the architecture of the model you are using
#     --model_file <path to models>/model_best.pt            \
#     --bert_model_file <path to models>/model_bert_best.pt  \
#     --bert_path <path to bert_config/vocab>  \
#     --result_path <where to place results>                 \
#     --data_path <path to db/jsonl/tables.jsonl>            \
#     --split <split>
#
# Results will be in a file called results_<split>.jsonl in the result_path.

import argparse
import io
import os
import re
import threading
import uuid
from typing import Union, Iterable, List

import numpy as np
import simplejson as json
import torch
from fastapi import FastAPI, Form, status, Response
from fastapi.middleware.cors import CORSMiddleware


import add_csv
import add_question
import annotate_ws
from sqlnet.dbengine import DBEngine
from sqlova.utils.utils_wikisql import (convert_pr_wvi_to_string,
                                        generate_sql_i, generate_sql_q,
                                        generate_sql_q_base, get_fields, get_g,
                                        get_g_wvi_corenlp, get_wemb_bert,
                                        load_wikisql_data, pred_sw_se,
                                        sort_and_generate_pr_w)
from train import construct_hyper_param, get_models
from wikisql.lib.query import Query

# Set up hyper parameters and paths
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_file", required=True,
#                     help='model file to use (e.g. model_best.pt)')
# parser.add_argument("--bert_model_file", required=True,
#                     help='bert model file to use (e.g. model_bert_best.pt)')
# parser.add_argument("--bert_path", required=True,
#                     help='path to bert files (bert_config*.json etc)')
# parser.add_argument("--data_path", required=True,
#                     help='path to *.jsonl and *.db files')
# parser.add_argument("--split", required=False,
#                     help='prefix of jsonl and db files (e.g. dev)')
# parser.add_argument("--result_path", required=True,
#                     help='directory in which to place results')
# args1 = construct_hyper_param(parser)

args = construct_hyper_param()

# handle_request = None

thread = None
status = "Loading sqlova model, please wait"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


# This is a stripped down version of the test() method in train.py - identical, except:
#   - does not attempt to measure accuracy and indeed does not expect the data to be labelled.
#   - saves plain text sql queries.
#
def predict(data_loader, data_table, model, model_bert, bert_config, tokenizer,
            max_seq_length,
            num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
            path_db=None, dset_name='test', columns=[], types=[], db_path="", table="trips"):
    model.eval()
    model_bert.eval()

    engine = DBEngine(db_path)

    results = []
    for iB, t in enumerate(data_loader):
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(
            t, data_table, no_hs_t=True, no_sql_t=True)
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)
        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(
                wemb_n, l_n, wemb_h, l_hpu, l_hs)
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(
                s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(
                pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            pr_sql_i = generate_sql_i(
                pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
            # Following variables are just for consistency with no-EG case.
            pr_wvi = None  # not used
            pr_wv_str = None
            pr_wv_str_wp = None


        pr_sql_q = generate_sql_q(pr_sql_i, tb)
        pr_sql_q_base = generate_sql_q_base(pr_sql_i, tb)
        print("FIST LOOP", results)
        results1 = {}
        for b, (pr_sql_i1, pr_sql_q1, pr_sql_q1_base) in enumerate(zip(pr_sql_i, pr_sql_q, pr_sql_q_base)):
            print("B",b)
            
            results1 = {"query": pr_sql_i1,
                        "table_id": tb[b]["id"],
                        "nlu": nlu[b],
                        "sql": pr_sql_q1,
                        "sql_with_params": pr_sql_q1_base}
            rr = engine.execute_query(tb[b]["id"],
                                      Query.from_dict(pr_sql_i1, ordered=True),
                                      columns=columns,
                                      types=types,
                                      table=table,
                                      lower=False)
            results1["answer"] = rr
            print(results1)
            # print("IN LOOP", results)
            # results.append(results1)

    return [results1]


BERT_PT_PATH = args.bert_path
path_save_for_evaluation = args.result_path

# Load pre-trained models
path_model_bert = args.bert_model_file
path_model = args.model_file
args.no_pretraining = True  # counterintuitive, but avoids loading unused models
model, model_bert, tokenizer, bert_config = get_models(
    args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)


def run_split(split, columns, types, db_path, table):
    # Load data
    dev_data, dev_table = load_wikisql_data(
        args.data_path, mode=split, toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)

    dev_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=dev_data,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    # Run prediction
    with torch.no_grad():
        results = predict(dev_loader,
                          dev_table,
                          model,
                          model_bert,
                          bert_config,
                          tokenizer,
                          args.max_seq_length,
                          args.num_target_layers,
                          detail=False,
                          path_db=args.data_path,
                          st_pos=0,
                          dset_name=split, EG=False, columns=columns, types=types, db_path=db_path, table=table)

    message = {
        "split": split,
        "result": results
    }
    return message


def serialize(o):
    if isinstance(o, np.int64):
        return int(o)


def encode_complex(obj) -> Union[int, float, Iterable, List[float], str]:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.float):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, complex):
            eturn [obj.real, obj.imag]
    return str(obj)

@app.post('/')
def question(response: Response, table_name: str = "trips", q: str = Form(...), debug: str = Form(...)):
    base = ""
    try:
        # filename = "data/test.csv"
        # Staging environment value: "postgres://postgres:postgres@localhost:5432/honda_dev"
        db_path = os.getenv("DB_URL")
        # if not 'csv' in request.files:
        #     raise Exception('please include a csv file')
        # if 'q' not in request.form:
        #     raise Exception(
        #         'please include a q parameter with a question in it')

        # csv = open(filename)
        # q = request.form['q']
        table_id = "trips_metadata"
        table_id = re.sub(r'\W+', '_', table_id)

        # table_name = "trips"

        record = add_csv.sql_to_json(
            table_id, 'tabled id blablbla', base + '.tables.jsonl')

        # Markup the questions
        add_question.question_to_json(table_id, q, base + '.jsonl')
        annotation = annotate_ws.annotate_example_ws(
            add_question.encode_question(table_id, q), record)

        # Create the standford nlp annotated tokenizer
        with open(base + '_tok.jsonl', 'a+') as fout:
            fout.write(json.dumps(annotation) + '\n')

        # Genereate the query, and run the result on SQL.
        message = run_split(
            base, record['header'], record['types'], db_path, table_name)
        code = 200
        
        if not debug:
            os.remove(base + '.jsonl')
            os.remove(base + '.tables.jsonl')
            os.remove(base + '_tok.jsonl')
            if 'result' in message:
                message = json.loads(json.dumps(message['result'][0], default=encode_complex))
                message['params'] = message['sql_with_params'][1]
                message['sql'] = message['sql_with_params'][0]
        
        return message
    except Exception as e:
        print(e)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}

    if debug:
        message['base'] = base

    


if args.split:
    message = run_split(args.split, [], [], "", "")
    json.dumps(message, indent=2, default=serialize)
    exit(0)


status = "Loading corenlp models, please wait"
annotate_ws.annotate('start up please')
