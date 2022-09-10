// sets the value num for number of clicks
var num;
chrome.storage.local.get("numClicks", value => {
  console.log(value.numClicks);
  num = value.numClicks;
});

// what to do when the window loads
window.onload = function () {
  newQuote();
  if (document.getElementById("numbers") != null) {
    var numbers = document.getElementById("numbers");
    numbers.innerHTML = num;
  }
}

// array of quotes
var quotes = ["We must all face the choice between what is right and what is easy.",
  "Always turn a negative situation into a positive situation.",
  "Success is no accident. It is hard work, perseverance, learning, studying, sacrifice, and most of all, love of what you are doing or learning to do.",
  "The future depends on what you do today.",
  "Winning doesn’t always mean being first. Winning means you’re doing better than you’ve done before.",
  "If you absolutely can’t stay positive, don’t go negative, just cruise neutral for a while until you can get back up.",
  "No one can make you feel inferior without your consent.",
  "The only man who never makes a mistake is the man who never does anything.",
  "True humility is not thinking less of yourself, it is thinking of yourself less.",
  "There is only one happiness in this life, to love and be loved.",
  "All students can learn and succeed, but not on the same day in the same way.",
  "It’s not what you’ve got, it’s what you use, that makes a difference in how your life turns out.",
  "No one is useless in this world who lightens the burden of it to anyone else.",
  "We should not praise people just for the sake of praising. We should honestly look for positive traits in the character and actions of the people around us.",
  "Reading is to the mind, as exercise is to the body."]

function newQuote() {
  var randomNum = Math.floor(Math.random() * (quotes.length));
  if (document.getElementById('quote') != null) {
    document.getElementById('quote').innerHTML = quotes[randomNum];
  }
}

// functional button counter
document.addEventListener('DOMContentLoaded', function () {
  if (document.getElementById("button") != null) {
    document.getElementById("button").addEventListener("click", buttonClick);
  }
});

var button = document.getElementById("button");

function buttonClick() {
  num += 1;
  var numbers = document.getElementById("numbers");
  chrome.storage.local.set({ "numClicks": num });
  numbers.innerHTML = num;
}

// resets the button counter
document.addEventListener('DOMContentLoaded', function () {
  if (document.getElementById("resetClicks") != null) {
    document.getElementById("resetClicks").addEventListener("click", reset);
  }
});

function reset() {
  num = 0;
  chrome.storage.local.set({ "numClicks": num });
  if (document.getElementById("numbers") != null) {
    var numbers = document.getElementById("numbers");
    numbers.innerHTML = 0;
  }
}

// links to music
document.addEventListener('DOMContentLoaded', function () {
  if (document.getElementById("lofi") != null) {
    document.getElementById("lofi").addEventListener("click", lofi);
  }
});

document.addEventListener('DOMContentLoaded', function () {
  if (document.getElementById("classical") != null) {
    document.getElementById("classical").addEventListener("click", classical);
  }
});

document.addEventListener('DOMContentLoaded', function () {
  if (document.getElementById("whiteNoise") != null) {
    document.getElementById("whiteNoise").addEventListener("click", whiteNoise);
  }
});

document.addEventListener('DOMContentLoaded', function () {
  if (document.getElementById("shuffle") != null) {
    document.getElementById("shuffle").addEventListener("click", shuffle);
  }
});

// random music
var music = ["https://music.youtube.com/playlist?list=RDCLAK5uy_mSaYvcoVvag_GrAc593WX3MoxTLvUXYT8",
  "https://music.youtube.com/playlist?list=RDCLAK5uy_mYa86YJNGqE4Jl5jG7DZP9NKdflw0hQL0",
  "https://music.youtube.com/playlist?list=PLnLNfS3gWKeGKqJl8muLi7HgcZRfR3p6Q",
  "https://music.youtube.com/playlist?list=RDCLAK5uy_kb7EBi6y3GrtJri4_ZH56Ms786DFEimbM"];



function lofi() {
  window.open("https://music.youtube.com/playlist?list=RDCLAK5uy_mSaYvcoVvag_GrAc593WX3MoxTLvUXYT8");
}

function classical() {
  window.open("https://music.youtube.com/playlist?list=RDCLAK5uy_mYa86YJNGqE4Jl5jG7DZP9NKdflw0hQL0");
}

function whiteNoise() {
  window.open("https://music.youtube.com/playlist?list=PLnLNfS3gWKeGKqJl8muLi7HgcZRfR3p6Q");
}
function shuffle() {
  window.open("https://music.youtube.com/playlist?list=RDCLAK5uy_kb7EBi6y3GrtJri4_ZH56Ms786DFEimbM");
};
