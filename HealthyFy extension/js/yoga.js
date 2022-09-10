window.onload = function(name){
    chooseYogaPose();
}

var yogaPoses = ["image/1.png", "image/2.png", "image/3.png", "image/4.png", "image/5.png", "image/6.png", "image/7.png", "image/8.png"]
var refreshid = null;
var timeGiven = 45;

function chooseYogaPose(){
    document.getElementById("yogaTimer").innerHTML = timeGiven;
    clearInterval(refreshid);
    yogaCounter(timeGiven);
    var randomNum = Math.floor(Math.random() * (yogaPoses.length));
    if (document.getElementById("yogaPose") != null) {
        document.getElementById("yogaPose").src = yogaPoses[randomNum];
    }
}

document.addEventListener('DOMContentLoaded', function () {
    if (document.getElementById("resetYogaPose") != null) {
      document.getElementById("resetYogaPose").addEventListener("click", chooseYogaPose);
    }
  });

  function yogaCounter(counter){
      refreshid =
      setInterval( function(){
          counter--;
          if (counter >= 0){
              document.getElementById("yogaTimer").innerHTML = counter;
          }
          if (counter == 0){
            document.getElementById("yogaTimer").innerHTML = "Complete!";
          }
      }, 1000);

  }