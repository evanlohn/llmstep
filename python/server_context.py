import argparse
import transformers
import torch
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from server import get_argparser, print_config

import vllm
import transformers
import os
import json

def load_vllm(model_name):
    print("Loading model...")
    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=1,
        dtype='bfloat16'
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    print("Done")
    return model, tokenizer


def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def _filter(texts, scores):
    texts_ = []
    scores_ = []
    for text, score in zip(texts, scores):
        if text.strip() in {"", "sorry", "admit"}:
            continue
        texts_.append(text)
        scores_.append(score)
    return texts_, scores_


def vllm_generate(
    model,
    tokenizer,
    prompt,
    temperatures,
    num_samples,
    max_new_tokens=128,
    stop=['</s>']
):
    texts, scores = [], []
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=stop,
            use_beam_search=temperature==0.0
        )
        outputs = model.generate([prompt], params, use_tqdm=False)
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '')
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


class LLMStepServer(HTTPServer):
    def __init__(
        self, model, tokenizer, generate_function, config
    ):
      self.model = model
      self.tokenizer = tokenizer
      self.generate_function = generate_function
      self.config = config

      address = (self.config['LLMSTEP_HOST'], self.config['LLMSTEP_PORT'])
      super().__init__(address, LLMStepRequestHandler)


class LLMStepRequestHandler(BaseHTTPRequestHandler):
    def process_request(self, tactic_state, prefix, context):
        prompts = self.server.config['LLMSTEP_PROMPTS']

        texts, scores = [], []
        for prompt_fn in prompts:
            prompt = prompt_fn(tactic_state, prefix, context)

            texts_, scores_ = self.server.generate_function(
                model=self.server.model,
                tokenizer=self.server.tokenizer,
                prompt=prompt,
                temperatures=self.server.config['LLMSTEP_TEMPERATURES'],
                num_samples=self.server.config['LLMSTEP_NUM_SAMPLES'],
                stop=['---', '\n']
            )
            texts.extend([prefix + text.strip() for text in texts_])
            scores.extend(scores_)
        texts, scores = _unique_sorted(texts, scores)
        texts, scores = _filter(texts, scores)

        response = {"suggestions": [(text, score) for text, score in zip(texts, scores)]}
        return response

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(post_data)
            result = self.process_request(
                data['tactic_state'],
                data['prefix'],
                data['context']
            )
            response = result
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            error_response = {'error': str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))


def llmstep_prompt1(tactic_state, prefix, context):
    prompt = """    
/- You are proving a theorem in Lean 4. You are given the following information: - The file contents up to the current tactic, inside [CTX]...[/CTX] - The current proof state, inside [STATE]...[/STATE] Your task is to generate the next tactic in the proof. Put the next tactic inside [TAC]...[/TAC] -/ [CTX] %s [/CTX] [STATE] %s [/STATE] [TAC] %s
""" % (context, tactic_state, prefix)
    return prompt

def get_config(args):
    config = {
        'LLMSTEP_MODEL': args.hf_model,
        'LLMSTEP_TEMPERATURES': args.temperatures,
        'LLMSTEP_NUM_SAMPLES': args.num_samples,
        'LLMSTEP_PROMPTS': [llmstep_prompt1],
        'LLMSTEP_HOST': os.environ.get('LLMSTEP_HOST', 'localhost'),
        'LLMSTEP_PORT': os.environ.get('LLMSTEP_PORT', 6000),
    }
    return config


if __name__ == '__main__':
    parser = get_argparser()
    parser.set_defaults(hf_model='l3lab/ntp-mathlib-context-deepseek-coder-1.3b')
    args = parser.parse_args()

    config = get_config(args)
    print_config(config)

    model, tokenizer = load_vllm(args.hf_model)

    httpd = LLMStepServer(
        model, tokenizer, vllm_generate, config
    )

    print('Server started')
    httpd.serve_forever()
