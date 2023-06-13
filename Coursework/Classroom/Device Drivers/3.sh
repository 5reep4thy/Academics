echo Enter url
read url

# curl to get the main html file
time curl $url > webpage.html

# wget to obtain all the necessary resources
wget -r -np -k $url 