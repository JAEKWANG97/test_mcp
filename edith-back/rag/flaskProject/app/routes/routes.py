from flask import Blueprint, request, jsonify
from app.services import portfolio
from app.services import logging
from quart import Quart, request, jsonify
import asyncio

import time

# Blueprint 생성
routes_bp = Blueprint('routes', __name__)

# 로거 설정
logger = logging.__get_logger()
app = Quart(__name__)

@routes_bp.route('/rag/health-check', methods=['GET'])
def health_check():
    logger.info('health check')
    logger.info('health check')
    return "I'm Alive!!!"


@routes_bp.route('/rag/portfolio', methods=['POST'])
def portfolio_make():
    try:
        # 요청 데이터 로깅
        data = await request.get_json()
        logger.info("Received request data: %s", data)

        # 데이터 추출
        user_id = data.get('userId')
        summaries = data.get('summaries', [])
        merge_request = data.get('mergeRequests', [])
        description = data.get('description', '')

        if description is None:
            logger.info("description is None")
            description = ''

        # 필드 값 로깅
        logger.info("Processing portfolio for userId: %s", user_id)
        logger.debug("Summaries: %s", summaries)
        logger.debug("Merge Requests: %s", merge_request)
        logger.debug("Description: %s", description)

        # 포트폴리오 생성 호출
        result = await portfolio.generate_portfolio(user_id, summaries, merge_request, description)
        logger.info("Portfolio creation result: %s", result)

        # 성공 응답
        if result:
            logger.info("Portfolio creation succeeded for userId: %s", user_id)
            return jsonify({'status': 'success', 'portfolio': result}), 200

        # 실패 응답
        logger.warning("Portfolio creation failed for userId: %s", user_id)
        return jsonify({'status': 'fail'}), 400

    except KeyError as e:
        logger.error("KeyError in request data: %s", e)
        return jsonify({'status': 'fail', 'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        logger.exception("Unexpected error occurred while processing portfolio")
        return jsonify({'status': 'fail', 'error': 'Internal server error'}), 500


@routes_bp.route('/rag/code-review', methods=['POST'])
def code_review():
    from app.services import reviewer

    data = request.get_json()
    url = data.get('url')
    token = data.get('token')
    projectId = data.get('projectId')
    branch = data.get('branch')
    changes = data.get('changes')
    logger.info(f'code review make = {projectId}')

    review, portfolio, techStack = reviewer.getCodeReview(url, token, projectId, branch, changes)
    if review and portfolio:
        return jsonify({'status': 'success', 'review': review, 'techStacks': techStack, 'summary': portfolio})
    else:
        return jsonify({'status': 'fail', 'message': '코드 리뷰 생성 중 오류가 발생했습니다.'}), 500


@routes_bp.route('/rag/advice', methods=['POST'])
def get_advice():
    from app.services import reviewer

    try:
        # 요청 방식 확인
        if request.method != 'POST':
            logger.warning("Invalid request method: %s", request.method)
            return jsonify({'status': 'fail', 'message': 'Invalid request method. Use POST.'}), 405

        # 요청 데이터 추출
        mr_summaries = request.get_json()
        if not isinstance(mr_summaries, list):
            logger.warning("Invalid data format. Expected a list.")
            return jsonify({'status': 'fail', 'message': 'Invalid data format. Expected a list.'}), 400

        # 로직 처리
        logger.info("Received MR Summaries: %s", mr_summaries)
        advice = reviewer.generate_advice(mr_summaries)

        # 성공 응답
        logger.info("Advice generated successfully.")
        return jsonify({'status': 'success', 'advice': advice}), 200

    except Exception as e:
        logger.exception("Error processing advice request: %s", e)
        return jsonify({'status': 'fail', 'message': 'Internal server error'}), 500
