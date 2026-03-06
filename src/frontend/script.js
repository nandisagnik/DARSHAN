const searchBtn = document.getElementById("search")
const resetBtn = document.getElementById("reset")
const questionInput = document.getElementById("question")
const answerBox = document.getElementById("answer")
const overlay = document.getElementById("thinkingOverlay")

const modal = document.getElementById("answerModal")
const modalAnswer = document.getElementById("modalAnswer")
const closeBtn = document.getElementById("closeAnswer")
const modalReset = document.getElementById("modalReset")


let selectedVideo = null

const videoCards = document.querySelectorAll(".videoCard")
const selectedLabel = document.getElementById("selectedVideo")

videoCards.forEach(card => {

card.addEventListener("click", () => {

if(card.classList.contains("placeholder")) return

/* remove previous selection */

videoCards.forEach(c => c.classList.remove("active"))

/* activate clicked card */

card.classList.add("active")

selectedVideo = card.dataset.video

selectedLabel.innerText = selectedVideo

})

})

/* force hide answer on startup */
answerBox.style.display = "none"
answerBox.innerText = ""

window.addEventListener("load", () => {

const intro = document.getElementById("introScreen")
const logo = document.getElementById("introLogo")
const text = document.getElementById("introTyping")
const main = document.getElementById("mainPage")

setTimeout(()=>{

logo.style.transform = "translateX(-40px)"

text.innerText = "Welcome to DARSHAN by 8oMATE"
text.classList.add("typing")

},500)

setTimeout(()=>{

intro.style.opacity="0"
intro.style.transition="opacity 0.6s"

setTimeout(()=>{
intro.style.display="none"
main.style.display="block"
},600)

},3200)

/* modal buttons */

closeBtn.onclick = () => {
modal.style.display = "none"
}

modalReset.onclick = () => {

questionInput.value = ""
modalAnswer.innerText = ""
modal.style.display = "none"
questionInput.focus()

}

})



async function runSearch(){

const question = questionInput.value.trim()

if(!question){
return
}

/* show thinking animation */

overlay.style.display = "flex"

try{

const response = await fetch("http://127.0.0.1:8000/ask",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
question: question,
video_id: selectedVideo
})
})

const data = await response.json()

/* hide overlay */

overlay.style.display = "none"

/* show modal answer */

modal.style.display = "flex"
modalAnswer.innerText = data.answer

}
catch(error){

overlay.style.display = "none"

modal.style.display = "flex"
modalAnswer.innerText = "Error connecting to backend."

}

}

/* click search */

searchBtn.onclick = runSearch

/* press enter */

questionInput.addEventListener("keydown", function(event){

if(event.key === "Enter"){
runSearch()
}

})

/* reset */

resetBtn.onclick = function(){

questionInput.value = ""
modalAnswer.innerText = ""
modal.style.display = "none"

overlay.style.display = "none"

/* clear selected video */

selectedVideo = null

videoCards.forEach(c => c.classList.remove("active"))

selectedLabel.innerText = "None"

questionInput.focus()

}