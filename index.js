const NUM_SAMPLES = 20
var base = "http://127.0.0.1:8000/Output/RandomSamples/"

var runs = [["ocean, alpha=10","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/batch_8/"],
            ["ocean, alpha=6","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/alpha_6/"],
            ["ocean, alpha=3","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/alpha_3/"],
            ["ocean, alpha=0","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/SinGAN/alpha_0/"],

            ["blackbird, alpha=10","dan-woje-desertmix-surfacev2/SinGAN/blackbird,heatmap/"],
            ["blackbird, alpha=6","dan-woje-desertmix-surfacev2/SinGAN/blackbird,heatmap,alpha_6/"],
            ["blackbird, alpha=0","dan-woje-desertmix-surfacev2/SinGAN/blackbird,heatmap,alpha_0/"],

            ["rabbit, alpha=10","walden-brush2-view1/SinGAN/rabbit,heatmap/"],
            ["rabbit, alpha=6","walden-brush2-view1/SinGAN/rabbit,heatmap,alpha_6/"],
            ["rabbit, alpha=0","walden-brush2-view1/SinGAN/rabbit,heatmap,alpha_0/"],

            ["tetra-fish, alpha=10","bilde_t_1/SinGAN/tetra_fish,heatmap/"],
            ["tetra-fish, alpha=6","bilde_t_1/SinGAN/tetra_fish,heatmap,alpha_6/"],

            ["ocean, random crop","181109-ocean-floor-acidity-sea-cs-327p_0c8e6758c89086c7dc520f4a8445fc08/random_crop/batch_8/"],
            ["blackbird, random crop","dan-woje-desertmix-surfacev2/random_crop/blackbird,heatmap/"],
            ["rabbit, random crop","walden-brush2-view1/random_crop/rabbit,heatmap/"],
            ["tetra-fish, random_crop","bilde_t_1/random_crop/tetra_fish,heatmap/"],
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




