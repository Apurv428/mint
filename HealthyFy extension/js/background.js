var waterRemind = {
    type: "basic",
    iconUrl: "./icons/water.png",
    title: "Drink Some Water!",
    message : "Click me once you're hydrated, the next reminder will be in 1 min!" 
}
var eyeRemind = {
    type: "basic",
    iconUrl: "./icons/icon48.png",
    title: "Rest your eyes!",
    message : "Click me once you're rested, the next reminder will be in 1 min!" 
}
var exerciseRemind = {
    type: "basic",
    iconUrl: "./icons/icon48.png",
    title: "Get up and move around!",
    message : "*VISIT THE Yoga Page in Auxiliary*\nClick me once you exercised, the next reminder will be in 1 min!" 
}

// Notification listener
chrome.notifications.onClicked.addListener(function (id) {
    if (id == "eye") {
        createEyeAlarm();
    } else if (id == "exerc") {
        createExerciseAlarm();
    } else if (id == "water") {
        createWaterAlarm();
    }
})

// Alarm Listener
chrome.alarms.onAlarm.addListener(function (alarmName){
    if (alarmName.name == "waterAlarm") {
        chrome.notifications.create("water", waterRemind, function (notificationID){})
    } else if (alarmName == "eyeAlarm") {
        chrome.notifications.create("eye", eyeRemind, function (notificationID){})
    } else if (alarmName == "exerciseAlarm") {
        chrome.notifications.create("exerc", exerciseRemind, function (notificationID){})
    }
    
});

// Alarm Creator
function createWaterAlarm(){
    console.log("water");
    chrome.alarms.create("waterAlarm", {delayInMinutes : 1.0});
}
function createEyeAlarm(){
    console.log("eye");
    chrome.alarms.create("eyeAlarm", {delayInMinutes : 1.0});
}
function createExerciseAlarm(){
    console.log("exerc");
    chrome.alarms.create("excerciseAlarm", {delayInMinutes : 1.0});
}

//Alarm Starter
chrome.alarms.create("eyeAlarm", {delayInMinutes : 1.0});
chrome.alarms.create("excerciseAlarm", {delayInMinutes : 1.0});
chrome.alarms.create("waterAlarm", {delayInMinutes : 1.0});



//notes appp

