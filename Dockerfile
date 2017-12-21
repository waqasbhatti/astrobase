FROM python:3.6-slim-stretch

# metadata
LABEL maintainer "Waqas Bhatti <waqas.afzal.bhatti@gmail.com>"

# install git, then pip install astrobase directly from git because we now have
# wheels for pyeebls. next, install ipython, and finally, download the JPL
# ephemerides
RUN apt-get update && apt-get -y install git --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir git+https://github.com/waqasbhatti/astrobase \
    && pip install --no-cache-dir ipython \
    && python -c "from astrobase import timeutils"

# setup the work directory
WORKDIR /astrobase

# the default command to run if invoked with docker run --rm -it or similar
# this just starts ipython
# to make local files in the current directory available to the docker
# container, use something like:
# docker run --rm -it -v `pwd`:/astrobase/work <container id>
CMD ["ipython"]
