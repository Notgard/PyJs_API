from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException
from starlette.status import HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND
from database.db import *


api_key_header = APIKeyHeader(name="access_token", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail="API KEY not found"
        )
    if api_key_header == verify_user(api_key_header):
        #print(api_key_header)
        #print(verify_user(api_key_header))
        return api_key_header   
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY"
        )
