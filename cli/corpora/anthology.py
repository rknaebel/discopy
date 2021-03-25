import datetime
import json
import re

import click
import spacy
from pybtex.database.input import bibtex
from pybtex.database.input.bibtex import PybtexSyntaxError

from discopy.data.loaders.raw import load_texts


def convert_tex(s):
    s = detex(s)
    s = s.replace('{\\"o}', 'ö')
    s = s.replace('{\\"a}', 'ä')
    s = s.replace('{\\"u}', 'ü')
    s = s.replace("{\\'e}", 'é')
    s = s.replace("{\\%}", '%')
    s = s.replace("{``}", '"')
    s = s.replace("``", '"')
    s = s.replace("{''}", '"')
    s = s.replace("''", '"')
    s = s.replace("{--}", '–')
    s = s.replace("--", '–')
    s = s.replace("{---}", '—')
    s = s.replace("---", '—')
    s = s.replace("{_}", '_')
    s = s.replace("- ", '')
    return s


def applyRegexps(text, listRegExp):
    """ Applies successively many regexps to a text"""
    # apply all the rules in the ruleset
    for element in listRegExp:
        left = element['left']
        right = element['right']
        r = re.compile(left)
        text = r.sub(right, text)
    return text


def detex(latexText):
    """Transform a latex text into a simple text"""
    # initialization
    regexps = []
    text = latexText
    # remove all the contents of the header, ie everything before the first occurence of "\begin{document}"
    text = re.sub(r"(?s).*?(\\begin\{document\})", "", text, 1)

    # remove comments
    regexps.append({r'left': r'([^\\])%.*', 'right': r'\1'})
    text = applyRegexps(text, regexps)
    regexps = []

    # - replace some LaTeX commands by the contents inside curly rackets
    to_reduce = [r'\\emph', r'\\textbf', r'\\textit', r'\\text', r'\\IEEEauthorblockA', r'\\IEEEauthorblockN',
                 r'\\author', r'\\caption', r'\\author', r'\\thanks']
    for tag in to_reduce:
        regexps.append({'left': tag + r'\{([^\}\{]*)\}', 'right': r'\1'})
    text = applyRegexps(text, regexps)
    regexps = []
    # - replace some LaTeX commands by the contents inside curly brackets and highlight these contents
    to_highlight = [r'\\part[\*]*', r'\\chapter[\*]*', r'\\section[\*]*', r'\\subsection[\*]*', r'\\subsubsection[\*]*',
                    r'\\paragraph[\*]*']

    # highlightment pattern: #--content--#
    for tag in to_highlight:
        regexps.append({'left': tag + r'\{([^\}\{]*)\}', 'right': r'\n#--\1--#\n'})
    # highlightment pattern: [content]
    to_highlight = [r'\\title', r'\\author', r'\\thanks', r'\\cite', r'\\ref']
    for tag in to_highlight:
        regexps.append({'left': tag + r'\{([^\}\{]*)\}', 'right': r'[\1]'})
    text = applyRegexps(text, regexps)
    regexps = []

    # remove LaTeX tags
    # - remove completely some LaTeX commands that take arguments
    to_remove = [r'\\maketitle', r'\\footnote', r'\\centering', r'\\IEEEpeerreviewmaketitle', r'\\includegraphics',
                 r'\\IEEEauthorrefmark', r'\\label', r'\\begin', r'\\end', r'\\big', r'\\right', r'\\left',
                 r'\\documentclass', r'\\usepackage', r'\\bibliographystyle', r'\\bibliography', r'\\cline',
                 r'\\multicolumn']

    # replace tag with options and argument by a single space
    for tag in to_remove:
        regexps.append({'left': tag + r'(\[[^\]]*\])*(\{[^\}\{]*\})*', 'right': r' '})
        # regexps.append({'left':tag+r'\{[^\}\{]*\}\[[^\]\[]*\]', 'right':r' '})
    text = applyRegexps(text, regexps)
    regexps = []

    # - replace some LaTeX commands by the contents inside curly rackets
    # replace some symbols by their ascii equivalent
    # - common symbols
    regexps.append({'left': r'\\eg(\{\})* *', 'right': r'e.g., '})
    regexps.append({'left': r'\\ldots', 'right': r'...'})
    regexps.append({'left': r'\\Rightarrow', 'right': r'=>'})
    regexps.append({'left': r'\\rightarrow', 'right': r'->'})
    regexps.append({'left': r'\\le', 'right': r'<='})
    regexps.append({'left': r'\\ge', 'right': r'>'})
    regexps.append({'left': r'\\_', 'right': r'_'})
    regexps.append({'left': r'\\\\', 'right': r'\n'})
    regexps.append({'left': r'~', 'right': r' '})
    regexps.append({'left': r'\\&', 'right': r'&'})
    regexps.append({'left': r'\\%', 'right': r'%'})
    regexps.append({'left': r'([^\\])&', 'right': r'\1\t'})
    regexps.append({'left': r'\\item', 'right': r'\t- '})
    regexps.append({'left': r'\\hline[ \t]*\\hline', 'right': r'============================================='})
    regexps.append({'left': r'[ \t]*\\hline', 'right': r'_____________________________________________'})
    # - special letters
    regexps.append({'left': r'\\\'{?\{e\}}?', 'right': r'é'})
    regexps.append({'left': r'\\`{?\{a\}}?', 'right': r'à'})
    regexps.append({'left': r'\\\'{?\{o\}}?', 'right': r'ó'})
    regexps.append({'left': r'\\\'{?\{a\}}?', 'right': r'á'})
    # keep untouched the contents of the equations
    regexps.append({'left': r'\$(.)\$', 'right': r'\1'})
    regexps.append({'left': r'\$([^\$]*)\$', 'right': r'\1'})
    # remove the equation symbols ($)
    regexps.append({'left': r'([^\\])\$', 'right': r'\1'})
    # correct spacing problems
    regexps.append({'left': r' +,', 'right': r','})
    regexps.append({'left': r' +', 'right': r' '})
    regexps.append({'left': r' +\)', 'right': r'\)'})
    regexps.append({'left': r'\( +', 'right': r'\('})
    regexps.append({'left': r' +\.', 'right': r'\.'})
    # remove lonely curly brackets
    regexps.append({'left': r'^([^\{]*)\}', 'right': r'\1'})
    regexps.append({'left': r'([^\\])\{([^\}]*)\}', 'right': r'\1\2'})
    regexps.append({'left': r'\\\{', 'right': r'\{'})
    regexps.append({'left': r'\\\}', 'right': r'\}'})
    # strip white space characters at end of line
    regexps.append({'left': r'[ \t]*\n', 'right': r'\n'})
    # remove consecutive blank lines
    regexps.append({'left': r'([ \t]*\n){3,}', 'right': r'\n'})
    # apply all those regexps
    text = applyRegexps(text, regexps)
    regexps = []
    # return the modified text
    return text


def bib_extract(s):
    try:
        parser = bibtex.Parser()
        bib_data = parser.parse_string(s)
        bib_data.entries.keys()
        for entry_key, entry in bib_data.entries.items():
            if 'abstract' not in entry.fields or 'language' in entry.fields and entry.fields['language'] != "English":
                continue
            if int(entry.fields['year']) < 2015:
                continue
            item = {
                'text': convert_tex(entry.fields['abstract']),
                'meta': {
                    'corpus': 'acl-anthologies',
                    'date': datetime.datetime.now().isoformat(),
                    'title': convert_tex(entry.fields['title']),
                    'year': int(entry.fields['year']),
                    'authors': [
                        convert_tex(str(p)) for p in entry.persons['author']
                    ] if 'author' in entry.persons else []
                }
            }
            return item
    except PybtexSyntaxError:
        return None


@click.command()
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(src, tgt):
    nlp = spacy.load('en')
    doc_i = 0
    content = []
    for line in src:
        content.append(line)
        if line == '}\n':
            entry = bib_extract(' '.join(content))
            if entry and not any(c in entry['text'] for c in '\\{}'):
                entry['docID'] = f'anthology_{doc_i:06}'
                parses = load_texts(texts=[entry['text']], nlp=nlp)[0].to_json()
                entry['text'] = parses['text']
                entry['sentences'] = parses['sentences']
                tgt.write(json.dumps(entry) + '\n')
            doc_i += 1
            content = []


if __name__ == '__main__':
    main()
