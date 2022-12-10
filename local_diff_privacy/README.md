Modified from [https://github.com/chamikara1986/LDPFL](https://github.com/chamikara1986/LDPFL)

# Requirements
* Docker (See installation instructions [here](https://docs.docker.com/engine/install/))
* Docker Compose (Instructions [here](https://docs.docker.com/compose/install/linux/))
* Data set (Install the SVHN dataset [here](http://ufldl.stanford.edu/housenumbers/))

# Run
To run the code, first build the docker image
`docker build -t ldpfl .`

Then, create and start the docker container using the compose file
`docker compose up`
