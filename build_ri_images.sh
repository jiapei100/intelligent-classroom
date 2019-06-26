#!/bin/bash
# ==============================================================================
# Copyright (C) <2018-2019> Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================


tag=$1

if [ -z "$tag" ]; then
    tag=latest
fi

BASEDIR=$(dirname "$0")
docker build . -t classroom-analytics:$tag 


#Pull grafana Image

docker pull grafana/grafana:6.1.6

#Pull InfluxDB Image

docker pull influxdb:1.7.6

#Create Volumes for data persistance
docker volume create --name=grafana-volume
docker volume create --name=influxdb-volume
docker volume create --name=classroom-analytics-volume

