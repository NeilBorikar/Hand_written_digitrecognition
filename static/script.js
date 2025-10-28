// AI Digit Recognition - Vercel Frontend Script

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;

// Initialize canvas
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth = 20;
ctx.lineCap = "round";

// Drawing functions
canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.beginPath();
});
canvas.addEventListener("mousemove", draw);

function draw(event) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

// Clear button
document.getElementById("clearBtn").addEventListener("click", () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").textContent =
    "Draw a digit and click Predict";
});

// Predict button
document.getElementById("predictBtn").addEventListener("click", async () => {
  const loader = document.getElementById("loader");
  const resultBox = document.getElementById("result");
  loader.classList.remove("hidden");
  resultBox.textContent = "Predicting...";

  // Convert canvas drawing to base64
  const image = canvas.toDataURL("image/png");

  try {
    // 🚀 Replace this URL with your actual backend endpoint
    const backendURL = "https://your-flask-backend.vercel.app/predict";

    const response = await fetch(backendURL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });

    const data = await response.json();
    loader.classList.add("hidden");
    resultBox.textContent = `Predicted Digit: ${data.digit}`;
  } catch (error) {
    loader.classList.add("hidden");
    resultBox.textContent = "Error connecting to backend!";
    console.error(error);
  }
});
