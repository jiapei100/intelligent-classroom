#!/bin/bash

classroom="9B"
camera="/data/sample.mp4"
influxIp="172.21.0.6"

while getopts 'r:c:i:h' OPTION; do
  case "$OPTION" in
    r)
      classroom="$OPTARG"
      ;;

    c)
      camera="$OPTARG"
      ;;

    i)
      influxIp="$OPTARG"
      ;;
    h)
      echo -e "script usage: $(basename $0) [-r Classroom Name] [-c Ip Camera Link] [-i InfluxDB Ip] \n" >&2
      exit 1
      ;;
    ?)
      echo "script usage: $(basename $0) [-r Classroom Name] [-c Ip Camera Link] [-i InfluxDB Ip] " >&2
      exit 1
      ;;
  esac
done

echo -e "\nThe Camera link provided for Classroom $classroom is $camera\n"

check_yes_no(){
while true; do
    read -p "Do you wish to install this program?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo -e "Please answer yes or no."\n;;
    esac
done
}

echo -e "This Script will install and run Classroom Analytics RI \n"
check_yes_no

sudo chmod a+x *.sh

#install Docker and Docker Compose

if [[ $(command -v docker) && $(command -v docker-compose) ]]; then
    echo -e "\nDocker and Docker-Compose are already installed\n"
  else
    sudo bash install_docker.sh
fi

# if the container image doesn't exist locally
if [[ "$(docker images -q classroom-analytics 2> /dev/null)" == "" ]]; then
    echo -e 'Reference Implementation Docker Image Does not exist : Building Images\n'
    bash build_ri_images.sh
fi
echo -e 'Starting Docker Containers For the Reference Implementation\n'
docker-compose up -d 
containerId=$(docker ps -qf "name=classroom-analytics")
echo -e "\nContainerId of the Classroom Analytics Container: $containerId\n"
echo -e 'Running the application .......\n'

xhost +
echo -e "\n"
echo -ne '###################                     (33%)\r'
sleep 10
echo -ne '######################################             (66%)\r'
sleep 10
echo -ne '#########################################################   (100%)\r'
echo -ne '\n'
echo -e "\n"

docker exec -it "$containerId" /bin/bash -c "source /opt/intel/openvino/bin/setupvars.sh && \
/root/inference_engine_samples_build/intel64/Release/classroom_analytics \
-pdc=/data/Retail/action_detection/pedestrian/rmnet_ssd/0165/dldt/person-detection-action-recognition-0005.xml \
-c=/data/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml \
-lrc=/data/Retail/object_attributes/landmarks_regression/0009/dldt/landmarks-regression-retail-0009.xml \
-pc=/data/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml \
-sc=/data/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.xml \
-frc=/data/Retail/object_reidentification/face/mobilenet_based/dldt/face-reidentification-retail-0095.xml \
-fgp=/opt/intel/openvino/inference_engine/samples/classroom_analytics/faces_gallery.json \
-i=\"$camera\" --influxip=$influxIp --cs=$classroom"





