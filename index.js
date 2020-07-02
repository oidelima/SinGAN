const NUM_SAMPLES = 20
var base = "http://127.0.0.1:8000/Output/RandomSamples/94547580-ivy-growing-on-the-forest-floor/"

var runs = [["SinGAN, batch_size=16, changing crop for alpha loss","SinGAN/batch_16_all_gpus/"],
            ["SinGAN, batch_size=16, fixed crop for alpha loss","SinGAN/batch_16_fixed_crop_for_alpha/"], 
            ["SinGAN, batch_size=16, fixed crop for alpha loss (only mask)","SinGAN/batch_16_fixed_crop_only_mask_part_for_alpha/"],
            ["SinGAN, batch_size=16, fullsized, changing crop for alpha loss","SinGAN/alpha_not_fixed/"],
            ["SinGAN, batch_size=16, fullsized, fixed crop for alpha loss (only mask)","SinGAN/batch_16_fullsized/"],
            ["batch_16,cropped,eye,changing_alpha", "SinGAN/batch_16,eye,changing_alpha/"],
            ["Random Crop, batch_size=16, fullsized","random_crop/batch_16_fullsized/"],
            ["Random Crop, batch_size=16, cropped","random_crop/batch_16_all_gpus/"]]

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




