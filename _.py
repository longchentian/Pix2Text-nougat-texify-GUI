import builtins
 
 
def wrapper(function):
    def _open(*args, **kw):
        """ 修改路径 """
        _args = list(args)
        _args[0] = __file__[:-4] + args[0]
        if kw.get('file'):
            kw['file'] = __file__[:-4] + kw['file']
        return function(*_args, **kw)
    return _open
 
 
setattr(builtins, 'open', wrapper(open))