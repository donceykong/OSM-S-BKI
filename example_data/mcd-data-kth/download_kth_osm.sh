# try another instance from the OSM wiki list
BASE="https://overpass.kumi.systems/api/interpreter"   # example instance
# BASE="https://overpass.openstreetmap.ru/api/interpreter"
# BASE="https://overpass.nchc.org.tw/api/interpreter"

bbox_s=59.34466; bbox_w=18.06195; bbox_n=59.35435; bbox_e=18.07865

curl -G "$BASE" \
  --data-urlencode "data=[out:xml][timeout:180];(node($bbox_s,$bbox_w,$bbox_n,$bbox_e);way($bbox_s,$bbox_w,$bbox_n,$bbox_e);relation($bbox_s,$bbox_w,$bbox_n,$bbox_e););out body;>;out skel qt;" \
  -o bbox.osm
