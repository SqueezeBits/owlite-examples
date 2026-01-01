FROM nvcr.io/nvidia/pytorch:24.07-py3

ARG GROUP_ID
ARG USER_ID


# RUN addgroup --gid $GROUP_ID owl
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID owlet


RUN python -m pip install owlite --extra-index-url https://pypi.squeezebits.com/


USER owlet