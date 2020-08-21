const NUM_SAMPLES = 20
var base = "http://127.0.0.1:8000/Output/RandomSamples"
var modes = ["SinGAN", "random_crop"]
var backgrounds = ["2081085051",  "dan-woje-desertmix-surfacev2"] //"charlottesville-7-view1",
var image_type = "full_fake";
var run_name = "no_l1,background,alpha_0"
var img_count = 0;
var urls = init_urls();
var start_time;
var image_url;
var img = document.getElementById("img");
img.onmousedown = GetCoordinates;
// console.log(urls)





document.getElementById("next").addEventListener("click", () => {
    load_images();
    start_time = new Date().getTime();
    //setTimeout(myFunction, 30000)
});

window.addEventListener('load', () => {
    load_images();
    start_time = new Date().getTime();
    //setTimeout(myFunction, 30000)
});

// function myFunction() {
//     alert('Out of time');
//   }



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


    document.getElementById("caption").innerHTML = image_url.split('/')[6]
    img.src = image_url
    document.getElementById("counter").innerHTML = "Image: " + String(img_count+1) + "/" + String(NUM_SAMPLES * modes.length * backgrounds.length)

    img_count += 1;


}

function FindPosition(oElement)
{
  if(typeof( oElement.offsetParent ) != "undefined")
  {
    for(var posX = 0, posY = 0; oElement; oElement = oElement.offsetParent)
    {
      posX += oElement.offsetLeft;
      posY += oElement.offsetTop;
    }
      return [ posX, posY ];
    }
    else
    {
      return [ oElement.x, oElement.y ];
    }
}

function GetCoordinates(e)
{
  var PosX = 0;
  var PosY = 0;
  var ImgPos;
  ImgPos = FindPosition(img);
  if (!e) var e = window.event;
  if (e.pageX || e.pageY)
  {
    PosX = e.pageX;
    PosY = e.pageY;
  }
  else if (e.clientX || e.clientY)
    {
      PosX = e.clientX + document.body.scrollLeft
        + document.documentElement.scrollLeft;
      PosY = e.clientY + document.body.scrollTop
        + document.documentElement.scrollTop;
    }
  PosX = PosX - ImgPos[0];
  PosY = PosY - ImgPos[1];

  var constraint_url = image_url.split("/")
  constraint_url[constraint_url.length-2] = "full_mask"
  constraint_url = constraint_url.join("/")
  console.log(constraint_url)
  img.src = constraint_url
  img.src = image_url


  console.log("X = ", PosX);
  console.log("Y = ", PosY);
  console.log("Image width", img.width )
  console.log("Image height", img.height)
}




// function change_image_type(){
//     image_type = document.querySelector('input[name="image_type"]:checked').value;
//     load_images()
// }

// function next_run(event){
    
//     if (event.target.id == "first-img") {
//         if (left_run < runs.length -1) left_run+= 1;
//         else left_run = 0
//     }
//     else if (event.target.id == "second-img"){
//         if (right_run < runs.length -1) right_run+= 1;
//         else right_run = 0
//     } 
//     load_images()
// }




