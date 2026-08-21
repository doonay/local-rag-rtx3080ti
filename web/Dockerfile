FROM nginx:1.30-alpine

COPY web/index.html /usr/share/nginx/html/index.html
COPY web/style.css /usr/share/nginx/html/style.css
COPY web/app.js /usr/share/nginx/html/app.js
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
