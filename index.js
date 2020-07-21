const NUM_SAMPLES = 20
var base = "http://127.0.0.1:8000/Output/RandomSamples/"

var runs = [["Extra layer (stride = 10)","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/discriminator_extra_layer/"],
            ["Extra layer (stride = 5)","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/extra_layer,stride_5/"], 
            ["Loss upweighted x 1","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/test,mult_1_00/"],
            ["Loss upweighted x 1.3828","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/test,mult_1_3828/"],
            ["Bird, extra layer (stride=20)","dan-woje-desertmix-surfacev2/SinGAN/test_avgpooling_2/"],
            ["Bird, average pooling ","dan-woje-desertmix-surfacev2/SinGAN/test_avgpooling/"],

            ["Random crop","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/random_crop/discriminator_extra_layer/"],
            ["Random crop","dan-woje-desertmix-surfacev2/random_crop/test_avgpooling/"]];



var curr_img = 0;
var left_run = 0;
var right_run = 6;
var image_type = "fake";


document.getElementById("prev").addEventListener("click", () => {
    if (curr_img > 0) {
        curr_img -= 1;
        load_images();
    }
});

document.getElementById("next").addEventListener("click", () => {
    if (curr_img < NUM_SAMPLES - 1) {
        curr_img += 1;
        load_images();
    }
});

window.addEventListener('load', () => {
    if (curr_img < NUM_SAMPLES - 1) {
        load_images();
    }
});

function load_images(){

    document.getElementById("caption-left").innerHTML = runs[left_run][0]
    document.getElementById("caption-right").innerHTML = runs[right_run][0]
    document.getElementById("first-img").src = base + runs[left_run][1] + image_type + "/" + String(curr_img) + ".png"
    document.getElementById("second-img").src = base + runs[right_run][1] + image_type + "/" + String(curr_img) + ".png"
    document.getElementById("counter").innerHTML = "Image: " + String(curr_img+1) + "/" + String(NUM_SAMPLES)

}


function change_image_type(){
    image_type = document.querySelector('input[name="image_type"]:checked').value;
    load_images()
}

function next_run(event){
    
    if (event.target.id == "first-img") {
        if (left_run < runs.length -1) left_run+= 1;
        else left_run = 0
    }
    else if (event.target.id == "second-img"){
        if (right_run < runs.length -1) right_run+= 1;
        else right_run = 0
    } 
    load_images()
}




