const NUM_SAMPLES = 20
var base = "http://127.0.0.1:8000/Output/RandomSamples/"

var runs = [["ivy, batch 16","94547580-ivy-growing-on-the-forest-floor/SinGAN/batch_16,alpha_fixed/"],
            ["ivy, batch 8","94547580-ivy-growing-on-the-forest-floor/SinGAN/batch_8/"],
            ["Ocean-floor","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/batch_8/"],
            ["coral yellow", "2081085051/SinGAN/test/"],
            ["bilde", "bilde_t_1/SinGAN/batch_8/"],
            ["coral_1", "coral_1/SinGAN/batch_8/"],
            ["desert", "dan-woje-desertmix-surfacev2/SinGAN/test/"],
            ["forest_1", "forest-floor-ivy-mud-1340969/SinGAN/batch_8,butterfly/"],
            ["forest_2", "ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea/SinGAN/batch_8/"],
            ["walden", "walden-brush2-view1/SinGAN/test/"],
            ["walden_log", "walen-log-view1/SinGAN/batch_8/"],
            ["woodchips", "woodchips-4-3/SinGAN/batch_8,frog/"],

            ["ivy, batch 16","94547580-ivy-growing-on-the-forest-floor/random_crop/batch_16,alpha_fixed/"],
            ["Ocean-floor","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/random_crop/batch_8/"],
            ["coral yellow", "2081085051/random_crop/test/"],
            ["bilde", "bilde_t_1/random_crop/batch_8/"],
            ["coral_1", "coral_1/random_crop/batch_8/"],
            ["desert", "dan-woje-desertmix-surfacev2/random_crop/test/"],
            ["forest_1", "forest-floor-ivy-mud-1340969/random_crop/batch_8,butterfly/"],
            ["forest_2", "ForestFloorDemo.0010-min-1920x1080-b51e92beb9b22acb25ba6a0508fcc7ea/random_crop/batch_8/"],
            ["walden", "walden-brush2-view1/random_crop/test/"],
            ["walden_log", "walen-log-view1/random_crop/batch_8/"],
            ["woodchips", "woodchips-4-3/random_crop/batch_8,frog/"]
        ]

var curr_img = 0;
var left_run = 0;
var right_run = 12;
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




