

FILE_PATH='20_newsgroups/alt.atheism/49960'
def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.

    Parameters
    ----------
    text : string
        The text from which to remove the signature block.
    """
    _before, _blankline, after = text.partition(r'\n\n')
    return after

with open(FILE_PATH,'rb') as f:
    text = str(f.read())
    text = strip_newsgroup_header(text)
    print(text)

