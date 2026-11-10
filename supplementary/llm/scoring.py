"""First-proposed-answer scoring used for the reported three-model study.

This preserves the archived extraction rules; later corrections never override
an earlier answer. New free-form responses may need a documented manual review.
"""
import re

word=re.compile(r"^[^A-Za-z0-9\u3400-\u9fff]*([A-Za-z]+(?:['’-][A-Za-z]+)*)")
preface=re.compile(r'^The word(?: that comes)?(?: immediately before\s+["“][^"”]+["”](?: in the phrase\s+["“][^"”]+["”])?)?\s+is\s+',re.I)
def unit(raw,lang):
 if lang=='zh':return raw.lstrip()[:1]
 m=word.match(raw);return m[1].lower() if m else ''
def proposed_answer(r):
 raw=(r['answer_raw'] or '').strip()
 if not raw:return '', 'empty_response'
 if r['language']=='zh':
  if raw in ['前一个'] or raw.startswith(('你提到的内容涉及','我无法提供')):
   return '', 'no_answer_statement'
  return unit(raw,'zh'),'direct_answer'
 # Ignore a complete restatement of the three inputs, never a partial answer.
 sep=r'(?:\s*,\s*|\s+)'
 echo=re.match('^'+sep.join(re.escape(r[k]) for k in ['a','b','c'])+r'(?=$|[\s.!?])',raw,re.I)
 echoed=bool(echo)
 if echo:raw=raw[echo.end():].lstrip(' \t\r\n.!?')
 if echoed and (not raw or raw.startswith('What word comes immediately before')):
  return '', 'input_or_question_echo_without_answer'
 m=preface.match(raw)
 if m:return unit(raw[m.end():],'en'), 'input_echo_then_answer_preface' if echoed else 'answer_preface'
 if raw.startswith(("I don't have enough context",'I apologize;','I cannot provide an answer','The phrase ')):
  return '', 'no_answer_statement'
 return unit(raw,'en'),'direct_answer'
def outcome(r,answer):
 return 'B' if answer==r['b'] else 'A' if answer==r['a'] else 'Q' if answer==r['c'] else 'Other'
