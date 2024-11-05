package com.ssafy.kubetest.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class kubetestController {

    @GetMapping("/test")
    public String kubetest() {
        return "kubetest";
    }
}
