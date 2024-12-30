const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("capture");
const resultDiv = document.getElementById("result");

// Access the user's webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        console.error("Error accessing camera: ", err);
    });

// Capture the image and send it for prediction
captureButton.addEventListener("click", () => {
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData.split(",")[1] }),
    })
        .then((response) => response.json())
        .then((data) => {
            resultDiv.innerHTML = `<p>Prediction: <strong>${data.prediction}</strong></p>`;
        })
        .catch((err) => {
            console.error("Error predicting: ", err);
            resultDiv.innerHTML = "<p>Error predicting the card. Please try again.</p>";
        });
});
