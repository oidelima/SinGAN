const NUM_SAMPLES = 3
var base = "http://127.0.0.1:8000/Output/RandomSamples"
var modes = ["SinGAN", "random_crop"]
var backgrounds = ["dan-woje-desertmix-surfacev2"] //"charlottesville-7-view1",
var image_type = "fake";
var run_name = "filter_fixed"
var img_count = 0;
var urls = init_urls();
var start_time;
var image_url;
var img = document.getElementById("img");
var img_mask = document.getElementById("img-mask");
img_mask.crossOrigin = "Anonymous";
var canvas = document.createElement('canvas');
var timer_duration = 5;
var history_ = [];
var stopTimer = false;

img.onmousedown = canvasClick;
// console.log(urls)





document.getElementById("next").addEventListener("click", () => {
    load_images();
    next_button = document.getElementById("next");
    next_button.style.display = "none"
    document.getElementById("console").className = "";
    document.getElementById("console").innerHTML = "";
    display = document.getElementById("timer");
    stopTimer = false;
    startTimer(timer_duration, display, outOfTime);
    //setTimeout(myFunction, 30000)
});

document.getElementById("start").addEventListener("click", () => {
  document.getElementById("intro-page").className = "d-none"
  document.getElementById("main-page").className = "d-inline"
  load_images();
  next_button = document.getElementById("next");
  next_button.style.display = "none"
  display = document.getElementById("timer");
  stopTimer = false;
  startTimer(timer_duration, display, outOfTime);
});

// window.addEventListener('load', () => {
//     load_images();
//     next_button = document.getElementById("next");
//     next_button.style.display = "none"
//     display = document.getElementById("timer");
//     stopTimer = false;
//     startTimer(timer_duration, display, outOfTime);
//     //setTimeout(myFunction, 30000)
// });



function init_urls(){
    var urls = [];
    for (var i = 0; i < modes.length; i++){
        for(var j = 0; j < backgrounds.length; j++){
            for (var k = 0; k < NUM_SAMPLES; k++){
                urls.push(base + "/" + backgrounds[j] + "/" + modes[i] + "/" +  run_name + "/" + image_type + "/" + String(k) + ".png")
            }
        }
    }
    return urls
    
}

function load_images(){

    var index =  Math.floor(Math.random() * urls.length);
    image_url = urls[index]
    urls.splice(index, 1)


    // document.getElementById("caption").innerHTML = image_url.split('/')[6]
    img.src = image_url

    constraint_url = image_url.split("/")
    constraint_url[constraint_url.length-2] = "mask_ind"
    constraint_url = constraint_url.join("/")
    img_mask.src = constraint_url
   
    document.getElementById("counter").innerHTML = "Image: " + String(img_count+1) + "/" + String(NUM_SAMPLES * modes.length * backgrounds.length)

    img_count += 1;


}

function canvasClick(e)
{
  canvas.width = img.width;
  canvas.height = img.height;
  canvas.getContext('2d').drawImage(img_mask, 0, 0, img.width, img.height);
  var pixelData = canvas.getContext('2d').getImageData(event.offsetX, event.offsetY, 1, 1).data;

  if (pixelData[0] > 0 && stopTimer == false){
    document.getElementById("console").innerHTML = "You found the animal. Press next for the following image";
    document.getElementById("console").className = "mb-3 alert alert-success";
    next_button = document.getElementById("next");
    next_button.style.display = "inline"
    stopTimer = true;
    var endTime = new Date();
    var timeDiff = endTime - startTime; //in ms
    timeDiff /= 1000;
    var algorithm = image_url.split('/')[6]
    history_.push([algorithm, timeDiff, image_url])

    if (img_count === NUM_SAMPLES*2){
      document.getElementById("main-page").className = "d-none"
      document.getElementById("finished-page").className = "d-inline"
    }
    console.log(history_)
    

  }
  console.log(pixelData)
}


function startTimer(duration, display,callback) {

  startTime = new Date();

  var timer = duration;

  var myInterval = setInterval(function() {

    if (stopTimer === true){
      clearInterval(myInterval);
      return
    }

    minutes = parseInt(timer / 60, 10)
    seconds = parseInt(timer % 60, 10);

    minutes = minutes < 10 ? "0" + minutes : minutes;
    seconds = seconds < 10 ? "0" + seconds : seconds;

    display.textContent = "Time remaining: " + minutes + ":" + seconds;

    if (--timer < 0) {
      timer = duration;
      
      // clear the interal
      clearInterval(myInterval);

      if (img_count === NUM_SAMPLES*2){
        document.getElementById("main-page").className = "d-none"
        document.getElementById("finished-page").className = "d-inline"
        return
      }

      // use the callback
      if(callback) {
          callback();
      }
      stopTimer = true;
    }
  }, 1000);
}

function outOfTime(){
  next_button = document.getElementById("next");
  next_button.style.display = "inline"
  document.getElementById("console").innerHTML = "You ran out of time. Press next for the following image";
  document.getElementById("console").className = "mb-3 alert alert-danger";
  var algorithm = image_url.split('/')[6]
  history_.push([algorithm, null, image_url])
  console.log(history_)

}




