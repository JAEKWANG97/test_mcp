package com.edith.developmentassistant.controller.dto.request;

import com.edith.developmentassistant.service.dto.request.RegisterProjectServiceRequest;
import java.util.List;

public record RegisterProjectRequest(
        Long id,
        String name,
        String contents,
        Long userId,
        List<String> branches
) {

    public RegisterProjectServiceRequest toServiceRequest() {
        return new RegisterProjectServiceRequest(
                this.id,
                this.name,
                this.contents,
                this.branches
        );
    }
}

