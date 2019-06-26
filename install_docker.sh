#!/bin/bash


echo "Installing Docker and Docker Compose"
apt-get update
wget -O - https://get.docker.com/ | bash
wget https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m) -O /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
