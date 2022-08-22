# Déploiement sur AWS EC2 

- installer nginx :

```bash
sudo apt install -y python3-pip nginx
```

- créer un fichier de configuration :

```bash
sudo vim /etc/nginx/sites-enabled/fastapi_nginx
```

- dans le fichier fastapi_nginx :
  
```bash
server {
        listen 80;
        server_name XX.XX.XX.XXX; # IPV4 publique de l'instance EC2
        location / {
                proxy_pass http://127.0.0.1:8000;
        }
}
```

- redémarrer nginx :

```bash
sudo service nginx restart
```

- cloner le repo github :

```bash
git clone https://github.com/AnodeGrindYo/SemanticSegmentationQuicknDirtyApp.git
```

- installer les librairies requises :
```bash
cd SemanticSegmentationQuicknDirtyApp
pip3 install -r requirements.txt
```

- lancer le serveur : 

```bash
python3 -m uvicorn main:app --reload
```

## Troubleshooting

### Problème avec l'installation de tensorflow

```bash
pip install tensorflow-cpu --no-cache-dir
```

### Problème avec l'installation d'opencv 
```bash
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
```

### Problème avec la taille des images
```bash
sudo vim /usr/local/nginx/conf/nginx.conf
```
Dans le contexte **http**, ou **location**, saisissez :
```bash
client_max_body_size 10M; # 10M, c'est juste un exemple
``` 

puis relancez nginx :
```bash
service nginx reload
```


### Autre problème

[stack**overflow**](https://stackoverflow.com/)

